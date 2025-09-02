# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch

from ..looper.dequantize_processor import DequantizeProcessor
from ..looper.eora_processor import EoraProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.input_cache import InputCache
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..looper.native_processor import NativeProcessor
from ..models import BaseGPTQModel
from ..models._const import SUPPORTS_MODULE_TYPES
from ..quantization.config import FORMAT
from ..models.loader import ModelLoader
from ..nn_modules.hooked_linear import HookedLinear, replace_module_with_hooked_legacy, replace_module_with_hooked_tree
from ..utils.logger import setup_logger
from ..utils.model import (find_modules, get_device, get_module, get_module_by_name_prefix,
                           get_moe_layer_modules, get_state_dict_for_save, make_quant, move_to, nested_move_to)
from ..utils.torch import (ALL_DEVICES, ALL_STREAMS, CPU, DEFAULT_BALANCE_STRATEGY,
                           HAS_CUDA, BalanceStrategy, device_next, device_next_reset,
                           torch_devices, torch_empty_cache, torch_streamCtx, torch_sync)
from ..version import __version__

log = setup_logger()

class ModuleLooper():
    def __init__(self, model: BaseGPTQModel, processors: List[LoopProcessor]):
        self.processors = processors
        self.gptq_model = model
        self.support_batch_quantize = model.support_batch_quantize
        self.lock = threading.Lock()

    def cache_inputs(self, layers, auto_gc, calibration_data, calibration_enable_gpu_cache, use_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device if calibration_enable_gpu_cache else CPU

        # TODO HookLinear add register_forward_pre_hook()
        def store_input_hook(module, args, kwargs):
            # Positional arguments.
            layer_input = []
            if kwargs.get("hidden_states") is not None:
                layer_input.append(move_to(kwargs["hidden_states"], device=data_device))
            else:
                # If hidden_states is not in kwargs, get it from the first positional argument
                # If error occurs here, check the model's modeling code
                layer_input.append(move_to(args[0], device=data_device))
                
            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs.get("attention_mask") is not None and str(type(module)) != "<class 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer'>":
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            raise ValueError

        # move layer to target device
        layers[0] = layers[0].to(self.gptq_model.quantize_config.device)
        ori_outside_layer_module_devices = {}
        for module_name in self.gptq_model.base_modules:
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to(module, cur_layer_device)
        # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        is_ovis = self.gptq_model.__class__.__name__ == "OvisGPTQ"
        self.gptq_model.pre_quantize_generate_hook_start()
        for example in calibration_data:
            for k, v in example.items():
                if str(type(layers[0])) == "<class 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer'>":
                    data_device = self.gptq_model.quantize_config.device
                else:
                    data_device = self.gptq_model.quantize_config.device if k == "pixel_values" else cur_layer_device
                if isinstance(v, list):
                    for index in range(len(v)):
                        if len(v[index].shape) == 1:
                            v[index] = v[index].unsqueeze(0)
                        v[index] = move_to(v[index].to(self.gptq_model.model.visual_tokenizer.dtype) if is_ovis else v[index],
                                                  device=data_device)
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, device=data_device)
            try:
                if str(type(layers[0])) == "<class 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer'>":
                    self.gptq_model.model.generate(**example, return_audio=False)
                else:
                    self.gptq_model.model(**example, use_cache=use_cache)
            except ValueError:
                pass
        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()
        move_to(layers[0], device=CPU)
        for module_name in self.gptq_model.base_modules:
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])
            if module is not None:
                move_to(module, device=ori_outside_layer_module_devices[module_name])
        if auto_gc:
            torch_empty_cache()
        return InputCache(layer_inputs=layer_inputs, layer_input_kwargs=layer_input_kwargs, position_ids=position_ids,
                          attention_masks=attention_masks)

    @torch.no_grad()
    def loop(self, auto_gc=True, calibration_enable_gpu_cache=True, buffered_fwd=False, **kwargs):
        if self.gptq_model.quantize_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError("quantization of `lm_head` layer with `tied_weights=True` model state is not supported. Please check model has `tied_weights=False`.")

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if get_module(self.gptq_model.model, key=self.gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = self.gptq_model.model.config.use_cache if hasattr(self.gptq_model.model.config, "use_cache") else False
        self.gptq_model.model.config.use_cache = False
        
        # Check if memory optimization is enabled via quantize config
        memory_optimization = any(p.qcfg.memory_optimization for p in self.processors if hasattr(p, 'qcfg'))
        
        if memory_optimization:
            # Memory optimization enabled - _layer_wise_info is guaranteed to exist
            layer_wise_info = self.gptq_model.model._layer_wise_info
            original_layers = layer_wise_info['original_layers']
            layers_prefix = layer_wise_info['layers_prefix']
            layer_count = layer_wise_info['layer_count']
            layers = [None] * layer_count  # Initialize with None placeholders
        else:
            # Standard loading (NO memory optimization)
            layers, layers_prefix = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.layers_node)
            layer_count = len(layers)
            original_layers = None  # No original_layers array needed

        for p_index, processor in enumerate(self.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, EoraProcessor) or\
                        (isinstance(processor, GPTQProcessor) and self.gptq_model.quantize_config.v2):
                    prev_processor = self.processors[p_index - 1]
                    processor.set_calibration_dataset(prev_processor.calibration_dataset)
                    # If calibration_dataset is None or Empty, the input_cache of the previous processor is used.
                    processor.receive_input_cache(copy.copy(prev_processor.inputs_cache))
                elif isinstance(processor, DequantizeProcessor):
                    # DequantizeProcessor does not perform any operations on dataset.
                    processor.set_calibration_dataset([])
                    processor.receive_input_cache(InputCache([], [], [], []))

                continue

            if memory_optimization:
                # In memory optimization mode, we need to get one layer to cache inputs
                # Since layers array contains None placeholders, we load the first layer temporarily
                if layer_count > 0:
                    temp_layer = ModelLoader._load_single_layer(
                        model_local_path=self.gptq_model.model_local_path,
                        config=self.gptq_model.model.config,
                        layers_prefix=layers_prefix,
                        layer_index=0,  # Load first layer for input caching
                        layer_modules=self.gptq_model.layer_modules,
                        **{"trust_remote_code": self.gptq_model.trust_remote_code}
                    )
                    input_cache = self.cache_inputs(layers=[temp_layer], auto_gc=auto_gc,
                                                    calibration_data=processor.calibration_dataset,
                                                    calibration_enable_gpu_cache=calibration_enable_gpu_cache,
                                                    use_cache=False)
                    # Clean up the temporary layer
                    del temp_layer
                else:
                    input_cache = InputCache([], [], [], [])
            else:
                # Standard mode - use layers array directly
                input_cache = self.cache_inputs(layers=layers, auto_gc=auto_gc,
                                                calibration_data=processor.calibration_dataset,
                                                calibration_enable_gpu_cache=calibration_enable_gpu_cache,
                                                use_cache=False)
            processor.receive_input_cache(input_cache)

        # release calibration_dataset
        for processor in self.processors:
            processor.release_calibration_dataset()

        layer_modules = self.gptq_model.layer_modules

        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        # dynamic expert layer index for model defs
        if self.gptq_model.dynamic_expert_index is not None:
            num_experts = getattr(self.gptq_model.model.config, self.gptq_model.dynamic_expert_index)
            layer_modules = get_moe_layer_modules(layer_modules=self.gptq_model.layer_modules,
                                                  num_experts=num_experts)

        layer_count = len(layers)
        quant_modules_pb = (log.pb(layer_count + 1 if self.gptq_model.quantize_config.lm_head else layer_count)
                            .manual()
                            .set(left_steps_offset=1))

        for processor in self.processors:
            processor.layer_count = layer_count
            processor.pb = quant_modules_pb

        shared_kv_cache_dict = {}

        # replace quantizable modules with hooked version
        if self.gptq_model.layers_modules_tree:
            replace_module_with_hooked_tree(self.gptq_model.model, self.gptq_model.layers_modules_tree, debug=False)
        else:
            replace_module_with_hooked_legacy(self.gptq_model.model)

        if self.gptq_model.quantize_config.lm_head:
            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module and isinstance(lm_head_module, torch.nn.Linear):
                hooked_lm_head = HookedLinear.from_linear(lm_head_module)
                module_path = self.gptq_model.lm_head.split('.')
                parent = self.gptq_model.model
                for part in module_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, module_path[-1], hooked_lm_head)


        for layer_index in quant_modules_pb:
            is_lm_head_module = layer_index >= layer_count

            if is_lm_head_module:
                quant_modules_pb.title("Quantizing lm_head").draw()
                module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            else:
                quant_modules_pb.title(f"Quantizing layer {layer_index} of {layer_count - 1}").draw()
                
                # Memory optimization enabled - use true layer-by-layer loading
                if memory_optimization:
                    # Load only the specific layer needed for quantization
                    layer = ModelLoader._load_single_layer(
                        model_local_path=self.gptq_model.model_local_path,
                        config=self.gptq_model.model.config,
                        layers_prefix=layers_prefix,
                        layer_index=layer_index,
                        layer_modules=self.gptq_model.layer_modules,
                        **{"trust_remote_code": self.gptq_model.trust_remote_code}
                    )
                    module = layer
                else:
                    # Standard loading (no memory optimization): load layer directly
                    module = layers[layer_index]

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue

            self.gptq_model.pre_quantize(module)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")

            for p_index, processor in enumerate(self.processors):
                processor.log_call_count = 0 # reset
                processor.collect_memory_info(layer_index)

                layer_inputs = processor.inputs_cache.layer_inputs
                if is_lm_head_module:
                    layer_inputs = self.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                processed_subset = {}

                modules = [[self.gptq_model.lm_head]] if is_lm_head_module else layer_modules

                # for NativeProcessor we process one time forward on all grouped module subsets
                if processor.fwd_all_modules_in_single_pass:
                    # merge all subsets into one
                    modules = [sum(modules, [])]

                for index, names in enumerate(modules):
                    subset = {}
                    for n in names:
                        if n in full:
                            subset[n] = full[n]
                        # some modules have layer_modules that are dynamic based on config
                        # ref: deepseek v2/v3/r1
                        elif self.gptq_model.layer_modules_strict:
                            raise ValueError(f"layer module item `{n}` not found in model, please check your model config.")

                    skipped_modules = []

                    for name in subset:
                        layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layers_prefix}.{layer_index}.{name}"

                        # gptq task is created and stored inside processor
                        if not isinstance(subset[name], NamedModule):
                            named_module = NamedModule(subset[name], name=name, full_name=layer_name,
                                                      layer_index=layer_index)
                            if isinstance(processor, EoraProcessor):
                                named_module.state.update({
                                    "wq": processor.quantized_weights[layer_name],
                                })
                                # TODO processor.release_quantized_weights()

                            subset[name] = named_module
                            full[name] = named_module

                        processor.preprocess(subset[name], buffered_fwd=buffered_fwd)
                        # some modules are skipped
                        if processor.is_skipped(subset[name]):
                            skipped_modules.append(name)

                    for name in skipped_modules:
                        subset.pop(name)

                    if len(subset) == 0:
                        continue

                    handle = []
                    # log.info(f"Subset = {subset}")
                    device_next_reset()

                    for name in subset:
                        m = subset[name]
                        m.module.target_device, m.module.target_device_stream = device_next()
                        # log.info(f"Loop name = {name}")
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = processor.pre_process_fwd_hook(name)
                        else:
                            # TODO FIXME: do we even need to hook into modules that are not quantizable?
                            assert (f"forward_hook missing for module name: `{name}`, layer name: {layer_name}")
                            handle.append(subset[name].register_forward_hook(processor.pre_process_fwd_hook(name)))

                    # ---- Start Pre-Quantized Forward ----
                    # logger.info(f"layer-{i}: Begin Forward() Pass")
                    fwd_start = time.time()

                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        # log.info(f"batch: {processor.num_batches}, j = {j}, layer_inputs = {layer_inputs}")
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device, stream=False))

                        mask = attention_masks[j]
                        layer_attention_mask = mask if mask is None else move_to(mask, device=cur_layer_device, stream=False)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                        layer_position_ids = (
                            None if not position_ids else move_to(position_ids[j], device=cur_layer_device, stream=False)
                        )

                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device, stream=False)

                        # sync above stream copies
                        #torch_sync(device=cur_layer_device)

                        # reuse_kv is a flag to reuse the kv cache, only for the hamba model
                        if hasattr(module, "reuse_kv"):
                            if module.reuse_kv:
                                additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(
                                    layer_index - 1)

                            layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input,
                                                                                                 **additional_layer_inputs)
                            if shared_kv_cache_dict.get(layer_index) is None:
                                shared_kv_cache_dict[layer_index] = layer_output[-1]
                        else:
                            layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input,
                                                                                  **additional_layer_inputs)
                        # For Native processor, we can update processor input here
                        # if second forward is not required, this/first forward output is captured as input for next loop
                        if not processor.fwd_after_process:
                            # after transformers 4.54, some model's DecodeLayer.forward() no longer returns tuple
                            if isinstance(layer_output, tuple):
                                layer_outputs.append([layer_output[0]])
                            else:
                                layer_outputs.append([layer_output])


                        del layer_input
                        del additional_layer_inputs

                    # Native processor does not need to run a second forward pass, the output of the first pass is
                    # directly saved and used as input for the next loop.
                    if not processor.fwd_after_process:
                        processor.receive_layer_inputs(layer_outputs)
                        del layer_outputs

                    fwd_end = time.time()
                    fwd_time = fwd_end - fwd_start

                    processor.set_fwd_time(fwd_time)

                    for h in handle:
                        h.remove()

                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None


                    # TODO FIXME: MoE modules forward() may not trigger if dataset is too small
                    # and moe gating logic does not trigger some moes
                    if isinstance(processor, GPTQProcessor):
                        moe_skip_modules = []
                        for name in subset :
                            if processor.tasks[name].fwd_counter == 0:
                                log.error(f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it.")
                                moe_skip_modules.append(name)

                        for name in moe_skip_modules:
                            subset.pop(name)
                    # ---- END Pre-Quantized Forward ----

                    # ---- Start Proceess Hook ----
                    if len(ALL_DEVICES) <= 1:
                        for name_index, name in enumerate(subset):
                            m = subset[name]
                            processor.process(module=m, auto_gc=auto_gc)
                            processed_subset[name] = m
                    else:
                        # TODO: there are threading/sync issues with streaming transfers
                        # for name in subset:
                        #     m = subset[name]
                        #     processor.pre_process_streaming(module=m)
                        #
                        # torch_sync()

                        # set to number of devices
                        max_workers = len(ALL_DEVICES) if DEFAULT_BALANCE_STRATEGY == BalanceStrategy.GPU else len(ALL_DEVICES) - 1
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = []
                            def process_module(name, m):
                                processor.process(module=m, auto_gc=auto_gc)
                                return name, m

                            for name in subset:
                                m = subset[name]
                                futures.append(executor.submit(
                                    process_module,
                                    name,
                                    m
                                ))

                            for future in futures:
                                name, m = future.result()
                                processed_subset[name] = m

                        torch_sync()
                    # ---- End Process Hook ----

                    if index == len(modules) - 1:
                        if auto_gc:
                            torch_empty_cache()

                is_last_module = layer_index == len(quant_modules_pb) - 1
                # Native processor does not need second forward pass after layer quantization
                # this is the second forward after process()
                if not is_last_module and processor.fwd_after_process:
                    layer_outputs = []
                    for j in range(processor.num_batches):
                        # assert weight
                        # if isinstance(processor, EoraProcessor):
                        #     for names in modules:
                        #         if n in names:
                        #             assert torch.equal(full[n].weight.data.cpu(), processed_subset[n].state["wq_ab"])
                        #             assert not torch.equal(full[n].weight.data.cpu(), processed_subset[n].state["wq"])
                        #             assert not torch.equal(processed_subset[n].state["wq_ab"], processed_subset[n].state["wq"])
                        #             full[n].weight.data.cuda()

                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device))

                        mask = attention_masks[j]
                        layer_attention_mask = mask if mask is None else move_to(mask, device=cur_layer_device)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}

                        layer_position_ids = None if not position_ids else move_to(position_ids[j], device=cur_layer_device)
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids

                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device)

                        if hasattr(module, "reuse_kv"):
                            if module.reuse_kv:
                                additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(layer_index - 1)

                        # log.info(f"MODULE Last forward: {module}")
                        module_output = None
                        if is_lm_head_module:
                            module_output = module(*layer_input)
                        else:
                            module_output = module(*layer_input, **additional_layer_inputs)

                        # after transformers 4.54, some model's DecodeLayer.forward() no longer returns tuple
                        if isinstance(module_output, tuple):
                            layer_output = module_output[0]
                        else:
                            layer_output = module_output
                            
                        layer_output = move_to(
                            layer_output,
                            device=cur_layer_device if calibration_enable_gpu_cache else CPU,
                            # stream=True,
                        )
                        
                        layer_outputs.append([layer_output])

                        del layer_input
                        del additional_layer_inputs
                        if processor.num_batches > 1 and j == processor.num_batches - 1:
                            if auto_gc:
                                torch_empty_cache()

                # TODO move to processor?
                if p_index == len(self.processors) - 1:
                    torch_sync()

                    if not is_lm_head_module:
                        quantized_layer = self.gptq_model.post_quantize(module)
                        
                        if memory_optimization:
                            # In memory optimization mode, save quantized layer to file and free RAM
                            self._pack_and_save_quantized_layer(layer_index, quantized_layer, layer_count)
                            # Free the quantized layer from memory
                            del quantized_layer
                            if auto_gc:
                                torch_empty_cache()
                            log.info(f"Saved and freed quantized layer {layer_index + 1}/{layer_count}")
                        else:
                            # Standard mode - store quantized layer back in layers array
                            layers[layer_index] = quantized_layer
                    else:
                        self.gptq_model.post_quantize(module)

                # This is second forward outputs captured for input of next loop
                # Native processor does not need second forward and already captured output from first forward
                if processor.fwd_after_process:
                    processor.clear_cache_data()
                    processor.receive_layer_inputs(layer_outputs)

                # if last processor, we need to call finalize in reverse
                if p_index == len(self.processors) - 1:
                    torch_sync()

                    for reverse_p in reversed(self.processors):
                        for name in processed_subset:
                            reverse_p.submodule_finalize(processed_subset[name])
                    del module
                    
                if auto_gc:
                    torch_empty_cache()

        total_log = {}

        for reverse_p in reversed(self.processors):
            if isinstance(reverse_p, GPTQProcessor):
                pass
                #logger.info(f"Quantization summary:\n{reverse_p.log}")
            elif isinstance(reverse_p, EoraProcessor):
                pass
                #logger.info(f"Eora summary:\n{reverse_p.log}")
            elif isinstance(reverse_p, DequantizeProcessor):
                # ignore log
                pass
            else:
                log.info(f"{reverse_p.name()} summary:\n{reverse_p.log}")

            processor_name = reverse_p.name()
            total_log[processor_name] = reverse_p.log
            if processor_name in ["gptq", "gptq v2"]:
                self.gptq_model.quant_log = reverse_p.log

            for module_log in reverse_p.log:
                log.info(module_log)
            reverse_p.log_plotly()

            reverse_p.finalize(model=self.gptq_model, **kwargs)

        self.gptq_model.model.config.use_cache = forward_pass_use_cache

        if auto_gc:
            torch_empty_cache()

        return total_log

    def _pack_quantized_layer(self, layer_index: int, quantized_layer):
        """
        Pack quantized layer based on single layer only.
        This method performs the packing logic similar to pack_model but for a single layer.
        Ensures proper packing by following the same steps as the finalize methods in processors.
        Handles both quantized modules and modules excluded from quantization via dynamic config.
        """
        import threadpoolctl as tctl
        
        layer_modules = self.gptq_model.layer_modules[0] if self.gptq_model.layer_modules else []
        
        # Step 1: Create quant_result dictionary similar to what processors collect
        quant_result = {}
        excluded_modules = []
        
        # Collect quantization results from the quantized layer and check for excluded modules
        for module_name in layer_modules:
            full_name = f"{self.gptq_model.layers_node}.{layer_index}.{module_name}"
            module = get_module(quantized_layer, module_name)
            
            if module is not None:
                # Check if this module is excluded from quantization via dynamic config
                if self.gptq_model.quantize_config.dynamic_get(full_name) == False:
                    excluded_modules.append(module_name)
                    log.info(f"Module {module_name} is excluded from quantization, will be saved in original state")
                    continue
                
                # Check if this module has quantization results in its state
                if hasattr(module, 'state') and module.state:
                    # Extract quantization results from module state
                    state = module.state
                    if "scale" in state and "zero" in state:
                        quant_result[full_name] = {
                            "scale": state["scale"],
                            "zero": state["zero"],
                            "g_idx": state.get("g_idx", None)
                        }
                        
                        # Handle QQQ-specific scale_extra if present
                        if "scale_extra" in state:
                            quant_result[full_name]["scale_extra"] = state["scale_extra"]
                        
                        log.info(f"Found quantization result for module {module_name} from module state")
                    else:
                        log.warning(f"Module {module_name} has state but missing scale/zero for packing")
                else:
                    log.warning(f"Module {module_name} has no state with quantization results")
            else:
                log.warning(f"Module {module_name} not found in quantized layer {layer_index}")
        
        # Step 2: Move the entire quantized layer to CPU for processing
        quantized_layer = quantized_layer.to(CPU)
        
        # Step 3: Handle excluded modules - save them in their original state
        if excluded_modules:
            log.info(f"Handling {len(excluded_modules)} excluded modules for layer {layer_index + 1}")
            for module_name in excluded_modules:
                # Get the original module from the model (not quantized)
                original_full_name = f"{self.gptq_model.layers_node}.{layer_index}.{module_name}"
                original_module = get_module(self.gptq_model.model, original_full_name)
                
                if original_module is not None:
                    # Clone the original module and move to CPU
                    original_module_copy = copy.deepcopy(original_module)
                    original_module_copy = original_module_copy.to(CPU)
                    
                    # Replace the module in the quantized layer with the original (unquantized) module
                    setattr(quantized_layer, module_name, original_module_copy)
                    log.info(f"Restored excluded module {module_name} to original state")
                else:
                    log.warning(f"Original module {module_name} not found for exclusion restoration")
        
        # Step 4: If we have quantization results, proceed with packing quantized modules
        if quant_result:
            log.info(f"Found {len(quant_result)} modules with quantization results for layer {layer_index + 1}")
            
            # Create a temporary model containing just this layer for proper quantization
            temp_model = torch.nn.Module()
            
            # Add all modules from this layer to the temp model
            for module_name in layer_modules:
                module = get_module(quantized_layer, module_name)
                if module is not None:
                    temp_module_name = f"layer_{module_name}"
                    setattr(temp_model, temp_module_name, module)
            
            # Step 5: Apply the same quantization logic as pack_model
            with tctl.threadpool_limits(limits=1):
                try:
                    # Create quantized linear modules using make_quant (similar to pack_model)
                    quant_linear_cls = make_quant(
                        model=temp_model,
                        quant_result=quant_result,
                        qcfg=self.gptq_model.quantize_config,
                        backend=self.gptq_model.qlinear_kernel if hasattr(self.gptq_model, 'qlinear_kernel') else self.gptq_model.backend,
                        lm_head_name=self.gptq_model.lm_head,
                        pack=True,
                    )
                    
                    log.info(f"Created quantized linear modules for layer {layer_index + 1}")
                    
                    # Step 6: Find the quantized modules and pack them
                    qModules = find_modules(temp_model, [quant_linear_cls])
                    
                    if len(qModules) == 0:
                        log.warning(f"No quantized modules found in layer {layer_index + 1}")
                        return quantized_layer
                    
                    # Step 7: Pack each module using the same logic as pack_module
                    for name, qmodule in qModules.items():
                        if name in quant_result:
                            r = quant_result[name]
                            scale = r["scale"]
                            zero = r["zero"]
                            g_idx = r.get("g_idx", None)
                            
                            # Get the original layer for packing
                            original_layer_name = name.replace("layer_", "")
                            original_layer = get_module(self.gptq_model.model, original_layer_name)
                            
                            if original_layer is not None:
                                # Move all tensors to CPU for packing
                                qmodule = qmodule.to(CPU)
                                original_layer = original_layer.to(CPU)
                                scale = scale.to(CPU)
                                zero = zero.to(CPU)
                                if g_idx is not None:
                                    g_idx = g_idx.to(CPU)
                                
                                # Handle QQQ special case
                                if hasattr(qmodule, 'QUANT_TYPE') and qmodule.QUANT_TYPE == "qqq":
                                    if "scale_extra" in r:
                                        scale_extra = r["scale_extra"].to(CPU)
                                        qmodule.pack(linear=original_layer, scales=scale, s_extra=scale_extra)
                                        log.info(f"Packed QQQ module {name} in layer {layer_index + 1}")
                                    else:
                                        log.warning(f"Missing scale_extra for QQQ module {name}")
                                else:
                                    # Standard GPTQ packing
                                    qmodule.pack(linear=original_layer, scales=scale, zeros=zero, g_idx=g_idx)
                                    log.info(f"Packed module {name} in layer {layer_index + 1}")
                            else:
                                log.warning(f"Original layer {original_layer_name} not found for module {name}")
                        else:
                            log.warning(f"No quantization result found for module {name}")
                    
                    # Step 8: Update the quantized layer with the properly packed modules
                    for module_name in layer_modules:
                        temp_module_name = f"layer_{module_name}"
                        temp_module = getattr(temp_model, temp_module_name, None)
                        if temp_module is not None:
                            # Update the module in the quantized layer
                            original_module = get_module(quantized_layer, module_name)
                            if original_module is not None:
                                setattr(quantized_layer, module_name, temp_module)
                    
                    log.info(f"Successfully packed layer {layer_index + 1}")
                    
                except Exception as e:
                    log.error(f"Failed to pack layer {layer_index + 1}: {e}")
                    # If packing fails, return the original quantized layer
                    log.info(f"Returning unpacked layer {layer_index + 1} as fallback")
        else:
            log.warning(f"No quantization results found for layer {layer_index + 1}")
        
        return quantized_layer
    
    def _save_quantized_layer(self, layer_index: int, quantized_layer, total_layers: int):
        """
        Save packed quantized layer in sharded format like model-00001-of-00028.safetensors
        and free RAM immediately after quantization.
        All weights of a layer are saved into a single file in final ready-for-inference state.
        Format conversion is applied to ensure saved layer files are final.
        """
        from safetensors.torch import save_file
        
        # Format layer number with leading zeros (5 digits for up to 99999 layers)
        layer_number = f"{layer_index + 1:05d}"  # +1 for 1-based indexing
        total_number = f"{total_layers:05d}"
        
        # Create shard filename for the entire layer - must match standard HF sharded format
        shard_filename = f"model-{layer_number}-of-{total_number}.safetensors"
        shard_file_path = os.path.join(self.gptq_model.model_local_path, shard_filename)
        
        # Get layer modules
        layer_modules = self.gptq_model.layer_modules[0] if self.gptq_model.layer_modules else []
        
        try:
            # Step 1: Create a temporary model containing just this layer for proper processing
            temp_model = torch.nn.Module()
            
            # Add all modules from this layer to the temp model
            for module_name in layer_modules:
                module = get_module(quantized_layer, module_name)
                if module is not None:
                    # Add module to temp model with prefixed name to avoid conflicts
                    temp_module_name = f"layer_{module_name}"
                    setattr(temp_model, temp_module_name, module)
                else:
                    log.warning(f"Module {module_name} not found in quantized layer {layer_index}")
            
            # Step 2: Apply format conversion if needed (same logic as save_quantized)
            if self.gptq_model.quantize_config.format == FORMAT.GPTQ:
                log.info(f"Applying GPTQ v1 format conversion for layer {layer_index + 1}")
                
                try:
                    from ..utils.model import convert_gptq_v2_to_v1_format
                    temp_model = convert_gptq_v2_to_v1_format(
                        temp_model,
                        quantize_config=self.gptq_model.quantize_config,
                        qlinear_kernel=self.gptq_model.qlinear_kernel
                    )
                    log.info(f"Successfully applied GPTQ v1 format conversion to layer {layer_index + 1}")
                except Exception as e:
                    log.error(f"Failed to apply GPTQ v1 format conversion for layer {layer_index + 1}: {e}")
                    log.warning(f"Proceeding with original format for layer {layer_index + 1}")
            
            # Step 3: Get the final state dict for saving (same logic as save_quantized)
            temp_model = temp_model.to(CPU)
            state_dict = get_state_dict_for_save(temp_model)
            
            # Step 4: Process state dict to ensure final inference state
            processed_state_dict = {}
            for temp_module_name, module in temp_model.named_modules():
                if not temp_module_name.startswith("layer_"):
                    continue
                    
                # Extract module name (remove "layer_" prefix)
                module_name = temp_module_name[6:]  # Remove "layer_" prefix
                
                # Get the original state dict for this module
                module_state_dict = {k: v for k, v in state_dict.items() if k.startswith(temp_module_name + ".")}
                
                # Remove the prefix and add proper module structure
                for k, v in module_state_dict.items():
                    new_key = f"{module_name}.{k[len(temp_module_name) + 1:]}"
                    processed_state_dict[new_key] = v
            
            # Step 5: Ensure all tensors are contiguous and on CPU (critical for inference)
            final_state_dict = {}
            for k, v in processed_state_dict.items():
                if isinstance(v, torch.Tensor):
                    final_state_dict[k] = v.clone().contiguous().to(CPU)
                else:
                    final_state_dict[k] = v
            
            # Step 6: Add proper metadata for loading (same as save_quantized)
            metadata = {
                "format": "pt",  # Required by Accelerate
                "layer_index": layer_index,
                "layer_modules": ",".join(layer_modules),
                "quant_method": self.gptq_model.quantize_config.quant_method,
                "bits": str(self.gptq_model.quantize_config.bits),
                "group_size": str(self.gptq_model.quantize_config.group_size),
                "sym": str(self.gptq_model.quantize_config.sym),
                "desc_act": str(self.gptq_model.quantize_config.desc_act),
                "total_modules": str(len(layer_modules)),
            }
            
            # Add quantization metadata (same as save_quantized)
            quantizers = [f"gptqmodel:{__version__}"]
            self.gptq_model.quantize_config.meta_set_versionable(
                key="quantizer",
                value=quantizers
            )
            metadata.update(self.gptq_model.quantize_config.meta)
            
            # Step 7: Save the layer in final inference-ready state
            if final_state_dict:
                save_file(final_state_dict, shard_file_path, metadata=metadata)
                log.info(f"Saved quantized layer {layer_index + 1}/{total_layers} to {shard_filename}")
                log.info(f"Layer {layer_index + 1} is in final ready-for-inference state")
            else:
                log.warning(f"No state dict found for layer {layer_index + 1}, skipping save")
            
        except Exception as e:
            log.error(f"Failed to save layer {layer_index}: {e}")
            raise

    def _pack_and_save_quantized_layer(self, layer_index: int, quantized_layer, total_layers: int):
        """
        Save quantized layer in sharded format like model-00001-of-00028.safetensors
        and free RAM immediately after quantization.
        All weights of a layer are saved into a single file.
        Modules are properly packed before saving.
        Format conversion is applied to ensure saved layer files are final.
        
        This method combines both packing and saving operations for backward compatibility.
        """
        # First pack the layer
        packed_layer = self._pack_quantized_layer(layer_index, quantized_layer)
        
        # Then save the packed layer
        self._save_quantized_layer(layer_index, packed_layer, total_layers)

