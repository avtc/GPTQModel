# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Isolated stage for capturing calibration inputs prior to quantization."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

import torch

from .. import DEVICE_THREAD_POOL
from ..looper.input_cache import InputCache
from ..nn_modules.hooked_linear import STOP_FORWARD_EXCEPTION, StopForward
from ..utils.ctx import ctx
from ..utils.device import get_device
from ..utils.looper_helpers import device_ctx
from ..utils.logger import setup_logger
from ..utils.model import get_module_by_name_prefix, move_to, nested_move_to
from ..utils.torch import CPU, META, torch_empty_cache

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .module_looper import ModuleLooper


class StageInputsCapture:
    """Capture layer inputs so processors can reuse cached activations."""

    def __init__(self, looper: ModuleLooper, logger=None) -> None:
        self.looper = looper
        self.gptq_model = looper.gptq_model
        self.logger = logger or setup_logger()

    def cache_inputs(
        self,
        layers: Sequence[torch.nn.Module],
        calibration_data: Iterable[Dict[str, torch.Tensor]],
        use_cache: bool,
    ) -> InputCache:
        layer_inputs: List[List[torch.Tensor]] = []
        attention_masks: List[torch.Tensor | None] = []
        position_ids: List[torch.Tensor] = []
        layer_input_kwargs: List[Dict[str, Any]] = []

        timer = getattr(self.gptq_model, "quant_region_timer", None)
        layer_label = None
        if layers:
            first_layer = layers[0]
            layer_label = getattr(first_layer, "full_name", None)
            if layer_label is None:
                layer_label = getattr(getattr(first_layer, "__class__", None), "__name__", None)
            if layer_label is None:
                layer_label = type(first_layer).__name__
            capture_source = f"cache_inputs:{layer_label}"
        else:
            capture_source = "cache_inputs"
        start_time = time.perf_counter() if timer else None

        try:
            calibration_batches = len(calibration_data)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            calibration_batches = None

        if calibration_batches is None:
            self.logger.info("ModuleLooper: capturing layer inputs (batch count unknown)")
        else:
            self.logger.info(
                "ModuleLooper: capturing layer inputs from %s calibration batches",
                calibration_batches,
            )

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device

        cache_forward_pb = None
        processed_rows = 0
        cache_total_batches = None
        if calibration_batches is not None and calibration_batches > 0:
            cache_total_batches = int(calibration_batches)
            cache_forward_pb = (
                self.logger.pb(range(max(cache_total_batches, 1)))
                .manual()
                .set(show_left_steps=False)
            )
            cache_title = (
                f"Forward cached inputs (Pre {layer_label})"
                if layer_label
                else "Forward cached inputs"
            )
            cache_forward_pb.title(cache_title).subtitle(
                f"Batch 0/{cache_total_batches}"
            ).draw()

        # VRAM DEBUG: Function to identify what's occupying VRAM
        def track_vram_objects(label: str):
            if data_device.type != "cuda":
                return

            import gc
            import sys

            # Scan all objects and find CUDA tensors
            tensor_objects = []
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        size_mb = obj.element_size() * obj.nelement() / 1024**2
                        if size_mb > 1:  # Only track > 1MB
                            # Try to find parent object
                            ref_count = sys.getrefcount(obj)
                            tensor_objects.append((size_mb, obj.shape, obj.device, type(obj).__name__, ref_count))
                except Exception:
                    pass

            # Group by tensor type and shape
            from collections import defaultdict
            groups = defaultdict(lambda: {"count": 0, "total_mb": 0.0, "shapes": []})

            for size_mb, shape, device, tensor_type, ref_count in tensor_objects:
                key = f"{tensor_type}"
                groups[key]["count"] += 1
                groups[key]["total_mb"] += size_mb
                if len(groups[key]["shapes"]) < 5:  # Limit shapes shown
                    groups[key]["shapes"].append(str(shape))

            self.logger.info(f"[VRAM-DEBUG] ===== {label} =====")
            total_tracked = 0.0
            for tensor_type, info in sorted(groups.items(), key=lambda x: x[1]["total_mb"], reverse=True):
                self.logger.info(f"[VRAM-DEBUG]   {tensor_type}: count={info['count']}, total={info['total_mb']:.1f}MB, shapes={info['shapes'][:3]}")
                total_tracked += info["total_mb"]
            self.logger.info(f"[VRAM-DEBUG]   Total tracked: {total_tracked:.1f}MB")

        # VRAM DEBUG: Track tensor allocations in InputCache
        total_cached_bytes = 0
        batch_count = [0]  # Use list to allow modification in closure

        def store_input_hook(module, args, kwargs):
            nonlocal total_cached_bytes
            batch_count[0] += 1

            layer_input: List[torch.Tensor] = []
            if kwargs.get("hidden_states") is not None:
                tensor = kwargs["hidden_states"]
                layer_input.append(move_to(tensor, device=data_device))
            else:
                tensor = args[0]
                layer_input.append(move_to(tensor, device=data_device))

            layer_inputs.append(layer_input)

            if kwargs.get("attention_mask") is not None:
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs: Dict[str, Any] = {}
            for (k, v) in kwargs.items():
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            # In normal repeating layer/sbuset early stop happens on the last module forward
            # but the first model input embedding call we use a simple model register forwar hook
            # and wait for the first instance this callback is called
            raise STOP_FORWARD_EXCEPTION

        if cur_layer_device == META:
            layers[0] = self.gptq_model.shell_module_materialize(
                target_submodule=layers[0],
                device=self.gptq_model.quantize_config.device,
            )
            cur_layer_device = self.gptq_model.quantize_config.device
        else:
            layers[0] = layers[0].to(self.gptq_model.quantize_config.device)

        ori_outside_layer_module_devices: Dict[str, torch.device] = {}
        base_modules_list = self.gptq_model.get_base_modules(self.gptq_model.model)
        self.logger.info(f"[VRAM-DEBUG] Base modules to materialize: {base_modules_list}")

        for module_name in base_modules_list:
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue

            m_device = get_device(module)
            self.logger.info(f"[VRAM-DEBUG] {module_name} device before materialize: {m_device}")
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device

            if module is not None:
                # VRAM FIX: If base module is on meta, materialize to CPU first then move to CUDA
                # This prevents extra +0.50GB VRAM allocation that occurs when meta modules
                # are materialized directly to CUDA for input capture.
                # By doing meta -> CPU -> CUDA, we match offload=False behavior where
                # base modules start on CPU and are moved to CUDA.
                if m_device == META:
                    self.logger.info(f"[VRAM-DEBUG] Materializing {module_name} from meta to CPU first (to avoid extra VRAM allocation)")
                    self.gptq_model.shell_module_materialize(
                        target_submodule=module,
                        device=CPU,
                    )
                    # Now move from CPU to CUDA (matching offload=False behavior)
                    self.logger.info(f"[VRAM-DEBUG] Moving {module_name} from CPU to CUDA")
                    module.to(cur_layer_device)
                else:
                    # Normal materialization to CUDA for non-meta modules
                    self.gptq_model.shell_module_materialize(
                        target_submodule=module,
                        device=cur_layer_device,
                    )

        # VRAM DEBUG: Check memory after base module materialization for input capture
        self.logger.info(f"[VRAM-DEBUG] ========== Input Capture: After Base Module Materialization ==========")
        torch_empty_cache()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            self.logger.info(f"[VRAM-DEBUG] cuda:{i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # Track what objects are in VRAM
        track_vram_objects("Objects in VRAM after base module materialization")

        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)

        is_ovis = self.gptq_model.__class__.__name__ == "OvisGPTQ"

        self.gptq_model.pre_quantize_generate_hook_start()

        try:
            for batch_index, example in enumerate(calibration_data, start=1):
                for k, v in example.items():
                    if self.gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                        data_device = self.gptq_model.quantize_config.device
                    else:
                        data_device = (
                            self.gptq_model.quantize_config.device
                            if k == "pixel_values"
                            else cur_layer_device
                        )
                    if isinstance(v, list):
                        for index in range(len(v)):
                            if len(v[index].shape) == 1:
                                v[index] = v[index].unsqueeze(0)
                            v[index] = move_to(
                                v[index].to(self.gptq_model.model.visual_tokenizer.dtype)
                                if is_ovis
                                else v[index],
                                device=data_device,
                            )
                    else:
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        example[k] = move_to(v, device=data_device)
                try:
                    if self.gptq_model.ATTENTION_MASKS_DTYPE is torch.long:
                        example["attention_mask"] = example["attention_mask"].long()

                    with ctx(
                        DEVICE_THREAD_POOL.read_lock(self.gptq_model.quantize_config.device),
                        device_ctx(self.gptq_model.quantize_config.device),
                    ):
                        if self.gptq_model.INPUT_EMBEDDING_EXTRA_ARGS:
                            self.gptq_model.model.generate(
                                **example,
                                **self.gptq_model.INPUT_EMBEDDING_EXTRA_ARGS,
                            )
                        elif is_ovis:
                            self.gptq_model.model.generate(inputs=example.pop("input_ids"), **example)
                        else:
                            self.gptq_model.model(**example, use_cache=use_cache)
                except StopForward:
                    pass
                finally:
                    processed_batches = batch_index
                    if cache_forward_pb is not None:
                        rows_for_batch = 0
                        if batch_index <= len(layer_inputs):
                            rows_for_batch = self.looper._batch_row_count(
                                layer_inputs[batch_index - 1]
                            )
                            if rows_for_batch <= 0:
                                rows_for_batch = 1
                        processed_rows += rows_for_batch
                        cache_forward_pb.current_iter_step = processed_batches
                        subtitle = f"Batch {processed_batches}/{cache_total_batches}"
                        if processed_rows > 0:
                            subtitle += f" rows {processed_rows}"
                        cache_forward_pb.subtitle(subtitle).draw()
        finally:
            if cache_forward_pb is not None:
                cache_forward_pb.close()

        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()

        # VRAM DEBUG: Track what's in VRAM immediately after input capture completes
        self.logger.info(f"[VRAM-DEBUG] ========== Input Capture: After Capture Loop (before InputCache) ==========")
        torch_empty_cache()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            self.logger.info(f"[VRAM-DEBUG] cuda:{i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        track_vram_objects("Objects in VRAM after input capture loop")

        # NOTE: First layer is NOT moved to meta after input capture
        # Moving it to meta breaks memory sharing between InputCache and the layer's forward buffers
        # causing InputCache to hold orphaned memory allocations that can't be freed
        # Both offload=True and offload=False now keep layer on cuda:0 for consistent memory behavior

        # VRAM DEBUG: Calculate exact InputCache size
        input_cache_size_mb = 0.0
        input_cache_tensors = 0
        if layer_inputs:
            for layer_input in layer_inputs:
                if layer_input:
                    for tensor in layer_input:
                        if torch.is_tensor(tensor) and tensor.is_cuda:
                            input_cache_size_mb += tensor.element_size() * tensor.nelement() / 1024**2
                            input_cache_tensors += 1

        self.logger.info(f"[VRAM-DEBUG] InputCache total size: {input_cache_size_mb:.2f}MB ({input_cache_tensors} tensors across {len(layer_inputs)} batches)")

        result = InputCache(
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
        )

        # VRAM DEBUG: Track VRAM after InputCache creation
        self.logger.info(f"[VRAM-DEBUG] ========== Input Capture: After InputCache Creation ==========")
        torch_empty_cache()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            self.logger.info(f"[VRAM-DEBUG] cuda:{i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        track_vram_objects("Objects in VRAM after InputCache creation")

        if timer is not None and start_time is not None:
            timer.record(
                "capture_inputs",
                time.perf_counter() - start_time,
                source=capture_source,
            )

        return result
