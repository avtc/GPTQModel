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
                           get_moe_layer_modules, get_state_dict_for_save, move_to, nested_move_to)
from ..utils.torch import (ALL_DEVICES, ALL_STREAMS, CPU, DEFAULT_BALANCE_STRATEGY,
                           HAS_CUDA, BalanceStrategy, device_next, device_next_reset,
                           torch_devices, torch_empty_cache, torch_streamCtx, torch_sync)

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
                            self._save_quantized_layer_to_file(layer_index, quantized_layer, layer_count)
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

    def _save_quantized_layer_to_file(self, layer_index: int, quantized_layer, total_layers: int):
        """
        Save quantized layer in sharded format like model-00001-of-00028.safetensors
        and free RAM immediately after quantization.
        All weights of a layer are saved into a single file.
        Modules are properly packed before saving.
        Format conversion is applied to ensure saved layer files are final.
        """
        from safetensors.torch import save_file
        import threadpoolctl as tctl
        
        # Format layer number with leading zeros (5 digits for up to 99999 layers)
        layer_number = f"{layer_index + 1:05d}"  # +1 for 1-based indexing
        total_number = f"{total_layers:05d}"
        
        # Create shard filename for the entire layer
        shard_filename = f"model-{layer_number}-of-{total_number}.safetensors"
        shard_file_path = os.path.join(self.gptq_model.model_local_path, shard_filename)
        
        # Accumulate all module states for this layer
        combined_state_dict = {}
        layer_modules = self.gptq_model.layer_modules[0] if self.gptq_model.layer_modules else []
        
        # Prepare quant_result for packing - this contains the quantization results
        quant_result = {}
        modules_to_pack = {}
        
        try:
            # First pass: collect quantization results and modules
            for module_name in layer_modules:
                # Get the quantized module
                full_name = f"{self.gptq_model.layers_node}.{layer_index}.{module_name}"
                module = get_module(quantized_layer, module_name)
                
                if module is not None:
                    # Check if this is a quantized linear module that needs packing
                    if hasattr(module, 'pack') and hasattr(module, 'quantize_config'):
                        # Store module for packing
                        modules_to_pack[module_name] = module
                        
                        # Try to get quantization results from module state if available
                        if hasattr(module, 'state') and module.state:
                            quant_result[module_name] = module.state
                            log.info(f"Found quantization result for module {module_name} from module state")
                    else:
                        # Regular module - get its state dict directly
                        if hasattr(module, 'state_dict'):
                            module_state = module.state_dict()
                            
                            # Add module prefix to avoid key conflicts
                            prefixed_state_dict = {}
                            for key, value in module_state.items():
                                prefixed_key = f"{module_name}.{key}"
                                prefixed_state_dict[prefixed_key] = value
                            
                            combined_state_dict.update(prefixed_state_dict)
                            log.info(f"Processed regular module {module_name} from layer {layer_index + 1}")
                        else:
                            log.warning(f"Module {module_name} has no state_dict method")
                else:
                    log.warning(f"Module {module_name} not found in quantized layer {layer_index}")
            
            # Second pass: pack modules that need packing
            if modules_to_pack and quant_result:
                # Limit pack() thread usage to avoid auto-parallizataion regression
                with tctl.threadpool_limits(limits=1):
                    log.info(f"Packing {len(modules_to_pack)} modules in layer {layer_index + 1}")
                    
                    for module_name, module in modules_to_pack.items():
                        try:
                            if module_name in quant_result:
                                # Get the original layer for packing
                                original_layer_name = f"{self.gptq_model.layers_node}.{layer_index}.{module_name}"
                                original_layer = get_module(self.gptq_model.model, original_layer_name)
                                
                                if original_layer is not None:
                                    # Pack the module with the original layer and quantization results
                                    scale = quant_result[module_name].get("scale")
                                    zero = quant_result[module_name].get("zero")
                                    g_idx = quant_result[module_name].get("g_idx")
                                    
                                    if scale is not None and zero is not None:
                                        # Move tensors to CPU for packing
                                        module = module.to(CPU)
                                        original_layer = original_layer.to(CPU)
                                        scale = scale.to(CPU)
                                        zero = zero.to(CPU)
                                        if g_idx is not None:
                                            g_idx = g_idx.to(CPU)
                                        
                                        # Perform the packing
                                        module.pack(linear=original_layer, scales=scale, zeros=zero, g_idx=g_idx)
                                        log.info(f"Packed module {module_name} in layer {layer_index + 1}")
                                        
                                        # Get the packed state dict
                                        packed_state = module.state_dict()
                                        
                                        # Add module prefix to avoid key conflicts
                                        prefixed_state_dict = {}
                                        for key, value in packed_state.items():
                                            prefixed_key = f"{module_name}.{key}"
                                            prefixed_state_dict[prefixed_key] = value
                                        
                                        combined_state_dict.update(prefixed_state_dict)
                                    else:
                                        log.warning(f"Missing scale or zero for module {module_name}, skipping pack")
                                else:
                                    log.warning(f"Original layer {original_layer_name} not found for module {module_name}")
                            else:
                                log.warning(f"No quantization result found for module {module_name}")
                        except Exception as e:
                            log.error(f"Failed to pack module {module_name}: {e}")
                            # Continue with unpacked module if packing fails
                            if hasattr(module, 'state_dict'):
                                try:
                                    module_state = module.state_dict()
                                    prefixed_state_dict = {}
                                    for key, value in module_state.items():
                                        prefixed_key = f"{module_name}.{key}"
                                        prefixed_state_dict[prefixed_key] = value
                                    combined_state_dict.update(prefixed_state_dict)
                                    log.info(f"Saved unpacked module {module_name} as fallback")
                                except Exception as e2:
                                    log.error(f"Failed to get state dict for module {module_name}: {e2}")
            
            # Apply format conversion before saving (similar to save_quantized logic)
            if self.gptq_model.quantize_config.format == FORMAT.GPTQ:
                log.info(f"Applying GPTQ v1 format conversion for layer {layer_index + 1}")
                
                # Create a temporary model containing just this layer for format conversion
                temp_model = torch.nn.Module()
                
                # Add all modules from this layer to the temp model
                for module_name in layer_modules:
                    module = get_module(quantized_layer, module_name)
                    if module is not None:
                        # Add module to temp model with prefixed name
                        temp_module_name = f"layer_{module_name}"
                        setattr(temp_model, temp_module_name, module)
                
                # Apply format conversion to the temporary model
                try:
                    from ..utils.model import convert_gptq_v2_to_v1_format
                    temp_model = convert_gptq_v2_to_v1_format(
                        temp_model,
                        quantize_config=self.gptq_model.quantize_config,
                        qlinear_kernel=self.gptq_model.qlinear_kernel
                    )
                    log.info(f"Successfully applied GPTQ v1 format conversion to layer {layer_index + 1}")
                    
                    # Update combined_state_dict with converted modules
                    converted_state_dict = {}
                    for module_name in layer_modules:
                        temp_module_name = f"layer_{module_name}"
                        temp_module = getattr(temp_model, temp_module_name, None)
                        if temp_module is not None and hasattr(temp_module, 'state_dict'):
                            module_state = temp_module.state_dict()
                            
                            # Add module prefix to avoid key conflicts
                            prefixed_state_dict = {}
                            for key, value in module_state.items():
                                prefixed_key = f"{module_name}.{key}"
                                prefixed_state_dict[prefixed_key] = value
                            
                            converted_state_dict.update(prefixed_state_dict)
                    
                    # Replace the original state dict with the converted one
                    combined_state_dict.update(converted_state_dict)
                    
                except Exception as e:
                    log.error(f"Failed to apply GPTQ v1 format conversion for layer {layer_index + 1}: {e}")
                    log.info(f"Proceeding with original format for layer {layer_index + 1}")
            
            # Add metadata for proper loading
            metadata = {
                "format": "pt",
                "layer_index": layer_index,
                "layer_modules": ",".join(layer_modules),
                "quant_method": self.gptq_model.quantize_config.quant_method,
                "bits": str(self.gptq_model.quantize_config.bits),
                "group_size": str(self.gptq_model.quantize_config.group_size),
                "sym": str(self.gptq_model.quantize_config.sym),
                "desc_act": str(self.gptq_model.quantize_config.desc_act),
                "total_modules": str(len(layer_modules)),
            }
            
            # Apply standard weight saving logic (similar to lines 319-437 in writer.py)
            if combined_state_dict:
                # Step 1: Ensure all tensors are on CPU
                cpu_state_dict = {}
                for k, v in combined_state_dict.items():
                    if isinstance(v, torch.Tensor):
                        cpu_state_dict[k] = v.to(CPU)
                    else:
                        cpu_state_dict[k] = v
                
                # Step 2: Apply get_state_dict_for_save logic to handle tensor aliasing
                # Since this is a layer, we need to be more careful about shared tensors
                # Create a temporary model to use get_state_dict_for_save
                temp_model = torch.nn.Module()
                for module_name in layer_modules:
                    # Get the processed module from combined_state_dict keys
                    module_prefix = f"{module_name}."
                    module_tensors = {k[len(module_prefix):]: v for k, v in cpu_state_dict.items() if k.startswith(module_prefix)}
                    
                    if module_tensors:
                        # Try to reconstruct the module
                        try:
                            # Get the original module structure
                            original_module = get_module(quantized_layer, module_name)
                            if original_module is not None:
                                # Create a new module with the same structure
                                reconstructed_module = original_module.__class__()
                                
                                # Set the tensors
                                for tensor_name, tensor_value in module_tensors.items():
                                    setattr(reconstructed_module, tensor_name, tensor_value)
                                
                                # Add to temp model with prefixed name
                                temp_module_name = f"layer_{module_name}"
                                setattr(temp_model, temp_module_name, reconstructed_module)
                        except Exception as e:
                            log.warning(f"Failed to reconstruct module {module_name}: {e}")
                            # Fall back to direct tensor handling
                            pass
                
                # Step 3: Use get_state_dict_for_save if we have a valid temp model
                if len(list(temp_model.named_modules())) > 0:
                    try:
                        filtered_state_dict = get_state_dict_for_save(temp_model)
                        log.info(f"Applied get_state_dict_for_save filtering for layer {layer_index + 1}")
                    except Exception as e:
                        log.warning(f"get_state_dict_for_save failed for layer {layer_index + 1}: {e}")
                        filtered_state_dict = cpu_state_dict
                else:
                    filtered_state_dict = cpu_state_dict
                
                # Step 4: Clone and make tensors contiguous (critical step)
                final_state_dict = {k: v.clone().contiguous() for k, v in filtered_state_dict.items()}
                
                # Step 5: Save using safetensors (metadata already contains "format": "pt")
                save_file(final_state_dict, shard_file_path, metadata=metadata)
                log.info(f"Saved quantized layer {layer_index + 1}/{total_layers} to {shard_filename}")
            else:
                log.warning(f"No state dict found for layer {layer_index + 1}, skipping save")
            
        except Exception as e:
            log.error(f"Failed to save layer {layer_index}: {e}")
            raise

