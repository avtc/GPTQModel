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

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.torch import HAS_CUDA, HAS_XPU, TORCH_GTE_28, device_next, torch_compile, torch_sync
from ..utils.importer import select_quant_linear
from ..utils.model import get_module
from .quantizer import HF_OPTIMUM, Quantizer

log = setup_logger()
lock = threading.Lock()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# TODO: is there a threading init bug in torch.linalg?
# bypass strange threading bug
if HAS_CUDA or HAS_XPU:
    tmp_eye = torch.eye(64, device="cuda" if HAS_CUDA else "xpu")
    torch.linalg.cholesky(tmp_eye)
    del tmp_eye


def get_number_of_rows_and_cols(layer: nn.Module):
    # return layer.weight.shape[0], np.prod(layer.weight.shape[1:])
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        # transformers.Conv1D: weight shape is (n_in, n_out)
        return layer.weight.shape[1], layer.weight.shape[0]
    else:
        # weight shape is (n_out, n_in)
        return layer.weight.shape[0], np.prod(layer.weight.shape[1:])


def _get_parent_and_layer_name(gptq_instance):
    """Helper function to get parent module and layer name for replacing modules"""
    if not hasattr(gptq_instance, 'layers_node') or not hasattr(gptq_instance, 'layer_index'):
        return None, gptq_instance.name
    
    # Split the name to get parent and layer name
    name_parts = gptq_instance.name.split('.')
    if len(name_parts) > 1:
        parent_name = '.'.join(name_parts[:-1])
        layer_name = name_parts[-1]
        return parent_name, layer_name
    else:
        return gptq_instance.layers_node, f"{gptq_instance.layers_node}.{gptq_instance.layer_index}"


class GPTQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None, model=None, layers_node=None, layer_index=None, backend=None, lm_head_name=None):
        # self.lock = threading.Lock()

        # self.num_tied_handles = 0
        # if qcfg.tied_gptq_handle is not None:
        #     qcfg.tied_gptq_handle.num_tied_handles += 1

        # Flags indicating issues
        # self.issue_zero_samples = False
        # self.issue_nan_hessian = False
        # self.issue_non_invertible = False

        # self.W = module.weight
        self.rows, self.columns = get_number_of_rows_and_cols(module)
        if isinstance(module, NamedModule):
            self.module = module.module
            self.name = module.name
        else:
            self.name = HF_OPTIMUM
            self.module = module
            # emulate NamedModule properties
            self.module.target_device, self.module.target_device_stream = device_next()

        self._validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg

        self.module_copy = None

        self.H = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd input buffer
        self.fwd_inputs_buffered = False
        self.fwd_inputs_buffered_data = []

        # fwd counter
        self.fwd_counter = 0

        # Memory optimization support
        self.temp_file_path = None
        self.temp_file_written = False
        
        # Additional attributes for memory optimization
        self.model = model
        self.layers_node = layers_node
        self.layer_index = layer_index
        self.backend = backend
        self.lm_head_name = lm_head_name

    @staticmethod
    def _validate_module(module):
        assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                   transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"

    # def has_hessian_issues(self) -> bool:
    #     return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def _mock_hessian_inverse(self, H: torch.Tensor):
        """Mock hessian inverse for fast testing"""
        damp = self.qcfg.damp_percent
        # Return identity matrix instead of complex inversion
        identity = torch.eye(H.shape[0], device=H.device)
        return identity, damp

    def _clone_module(self, copy=True, device: torch.device = None):
        if not device:
            device = self.module.weight.data.device

        clone = self.module.weight.data.to(copy=copy, device=device)

        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        return clone.float()

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        self.fwd_counter += 1

        if self.fwd_inputs_buffered:
            # with torch_streamCtx(self.module.target_device_stream):
            #     self.fwd_inputs_buffered_data.append(inp.to(device=self.module.target_device, non_blocking=True))
            self.fwd_inputs_buffered_data.append(inp.to(device=self.module.target_device, non_blocking=False))
        else:
            self.process_batch(inp)

    def process_batch(self, inp: torch.Tensor):
        inp = inp.to(device=self.module.target_device, dtype=torch.float32)

        # input reshaping
        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            reshaped_inp = inp.reshape(-1, inp.shape[-1])
        else:
            if isinstance(self.module, nn.Conv1d):
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    1,
                )
                unfold = nn.Unfold(
                    self.module.kernel_size + (1,),
                    dilation=self.module.dilation + (1,),
                    padding=self.module.padding + (0,),
                    stride=self.module.stride + (1,),
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            else:
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                unfold = nn.Unfold(
                    self.module.kernel_size,
                    dilation=self.module.dilation,
                    padding=self.module.padding,
                    stride=self.module.stride,
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            reshaped_inp = reshaped_inp.transpose(1, 2).flatten(0, 1)

        batch_token_size = reshaped_inp.shape[0]

        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns),
                                 dtype=torch.float32,
                                 device=reshaped_inp.device)

        # moe model may receive an empty batch, return early
        if batch_token_size == 0:
            return batch_token_size, reshaped_inp, 0, 0

        beta = self.nsamples / (self.nsamples + batch_token_size)
        alpha = 2.0 / (self.nsamples + batch_token_size)

        self.H.addmm_(reshaped_inp.T, reshaped_inp, beta=beta, alpha=alpha)

        # update number of collected samples
        self.nsamples += batch_token_size

        # inp returned here is flattened/reshaped original inp
        # return batch_token_size, reshaped_inp, alpha, beta
        del batch_token_size, reshaped_inp, alpha, beta

    # FIXME, optimum needs fasterquant, we need to remove it
    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    @torch.inference_mode()
    def hessian_inverse(self, H: torch.Tensor):

        damp = self.qcfg.damp_percent
        diag = torch.arange(self.columns, device=H.device)
        mean = torch.mean(torch.diag(H))
        while 0 < damp < 1:
            try:
                H2 = H.clone()
                H2[diag, diag] += damp * mean
                # TODO call to torch.linalg is not threadsafe? Porque no? Esta muy mal.
                H2 = torch.linalg.cholesky(H2)
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(H2), upper=True)
                del H, H2
                break
            except torch._C._LinAlgError as e:
                if self.qcfg.damp_auto_increment != 0:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Current `damp_percent = {damp:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`")
                    damp += self.qcfg.damp_auto_increment
                else:
                    log.warn(
                        "Quantization: Module `{self.name}` -> Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp_percent:.5f}`")
                    raise e

        if not (0 < damp < 1):
            log.error(
                f"Quantization: Module `{self.name}` -> `damp_percent` must between 0 and 1. current is {damp}. Module cannot be correctly processed.")
            # raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp}")
            return None, 1.0

        return Hinv, damp

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        # Temporarily disable torch.compile due to compatibility issues with torch 2.8
        # Will re-enable once the issue is fixed
        if not TORCH_GTE_28:
            self.hessian_inverse = torch_compile(self.hessian_inverse)

        # Mock heavy computations
        if hasattr(self.qcfg, 'mock_quantization') and self.qcfg.mock_quantization:
            # Use simplified hessian inverse (identity matrix)
            self.hessian_inverse = self._mock_hessian_inverse

        # process buffered inputs
        if len(self.fwd_inputs_buffered_data) > 0:
            torch_sync(device=self.module.target_device)

            for inp in self.fwd_inputs_buffered_data:
                self.process_batch(inp)

            # release buffer
            del self.fwd_inputs_buffered_data

        # if self.device.type not in ["mps", "cpu"]:
        #     self.module.weight.data = self.module.weight.data.cpu()

        # TODO: waiting for pytorch implementation of ops for MPS
        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

        if self.module_copy is None:
            # log.info("copy W to cuda_1")
            W = self._clone_module(device=self.module.target_device)
        else:
            W = self.module_copy.to(device=self.module.target_device)
            del self.module_copy

        self.quantizer.find_params(W, weight=True)

        H = self.H.to(device=self.module.target_device)
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if self.qcfg.static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, self.qcfg.group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i: (i + self.qcfg.group_size)], weight=True)

                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if self.qcfg.desc_act:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        if hasattr(self.qcfg, "hyb_act") and self.qcfg.hyb_act and not self.qcfg.desc_act:
            from .gar import compose_final_perm, compute_global_perm, compute_local_perms
            local_perms = compute_local_perms(torch.diag(H), self.qcfg.group_size)
            global_perm = compute_global_perm(torch.diag(H), self.qcfg.group_size)
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            W = W[:, final_perm]
            H = H[final_perm][:, final_perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        Hinv, damp = self.hessian_inverse(H)

        # Use simplified loop when mock_quantization is active
        if hasattr(self.qcfg, 'mock_quantization') and self.qcfg.mock_quantization:
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                # Clone the weights like the original code to maintain device/dtype consistency
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)

                # Handle group quantization parameters efficiently (similar to original)
                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        # Find parameters for entire groups at once (optimized)
                        group_start_cols = list(range(i1, i2, self.qcfg.group_size))
                        for group_start in group_start_cols:
                            group_end = min(group_start + self.qcfg.group_size, self.columns)
                            if group_start < group_end:
                                self.quantizer.find_params(W[:, group_start:group_end], weight=True)
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                    else:
                        # Static groups - use pre-computed groups
                        for i in range(count):
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]

                    # Vectorized quantization for the entire block (major optimization)
                    if len(scale) > 0 and len(zero) > 0:
                        # Use latest scale and zero for the entire block
                        latest_scale = scale[-1]
                        latest_zero = zero[-1]

                        # Vectorized quantization using broadcasting
                        # Reshape scales and zeros to match block dimensions
                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)

                        # Apply quantization formula using the cloned weights W1
                        maxq_val = 2 ** self.qcfg.bits - 1
                        if self.qcfg.sym:
                            # Symmetric quantization: Q = scale * clamp(round(x/scale), -maxq/2, maxq/2)
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            # Asymmetric quantization: Q = scale * (clamp(round(x/scale) + zero, 0, maxq) - zero)
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        # Fallback to individual quantization if no scale/zero available
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q
                else:
                    # No grouping - vectorized quantization for entire block
                    maxq_val = 2 ** self.qcfg.bits - 1
                    if hasattr(self.quantizer, 'scale') and hasattr(self.quantizer, 'zero'):
                        latest_scale = self.quantizer.scale
                        latest_zero = self.quantizer.zero

                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)

                        if self.qcfg.sym:
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        # Fallback to individual quantization
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q

                Q[:, i1:i2] = Q1
        else:
            # Original heavy loop for normal quantization
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)

                if Hinv is not None:
                    Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    if Hinv is not None:
                        d = Hinv1[i, i]

                    if self.qcfg.group_size != -1:
                        if not self.qcfg.static_groups:
                            if (i1 + i) % self.qcfg.group_size == 0:
                                self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + self.qcfg.group_size)], weight=True)

                            if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]

                            self.quantizer = groups[idx // self.qcfg.group_size]

                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    if Hinv is not None:
                        Losses1[:, i] = (w - q) ** 2 / d**2
                        err1 = (w - q) / d
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                if Hinv is not None:
                    Losses[:, i1:i2] = Losses1 / 2
                    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # TODO: why is there a torch_sync here? There are no streaming ops here?
        # torch_sync(device=self.module.target_device)

        if Hinv is not None:
            del Hinv
            if self.nsamples != 0:
                avg_loss = torch.sum(Losses).item() / self.nsamples

                if math.isnan(avg_loss):
                    print("Losses sum item:", torch.sum(Losses).item())
                    raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`")
            else:
                log.warn(f"Quantization: `{self.name}` is not activated due to model inference logic (MoE)")
                avg_loss = 999999999
        else:
            avg_loss = 999999999

        del Losses

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]

        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if self.qcfg.desc_act:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if hasattr(self.qcfg, "hyb_act") and self.qcfg.hyb_act and not self.qcfg.desc_act:
            from .gar import invert_perm
            inv_final = invert_perm(final_perm)
            Q = Q[:, inv_final]
            inv_global_perm = invert_perm(global_perm)
            inv_global_perm_list = inv_global_perm.tolist()
            temp_scale = [scale[i] for i in inv_global_perm_list]
            scale = temp_scale
            temp_zero = [zero[i] for i in inv_global_perm_list]
            zero = temp_zero

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        # Ensure Q is on the same device as the original module weight before type conversion
        if Q.device != self.module.weight.data.device:
            Q = Q.to(device=self.module.weight.data.device)

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
        else:
            Q = Q.type_as(self.module.weight.data)

        # Q = Q.to(device=use_device)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        duration = time.time() - start

        # Memory optimization: write quantized weights to file and clear memory
        if self.qcfg.memory_optimization:
            # For memory optimization, we'll pack directly to safetensors format
            # This eliminates the need for temporary files and reload_layer calls
            safet_file = self.pack_layer_to_safetensors(Q, scale, zero, g_idx, "quantized_layers")
            log.info(f"Memory optimization: Layer {self.name} quantized and packed directly to {safet_file}")
            # Return None for Q to indicate it's saved to disk
            return None, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def pack_layer_to_safetensors(self, q, scale, zero, g_idx, save_dir):
        """Pack quantized layer directly to final quantized modules for memory optimization"""
        if not self.qcfg.memory_optimization:
            return None
            
        import os
        import safetensors
        import torch
        
        # Create a temporary directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a unique filename for this layer
        layer_name = self.name.replace('.', '_').replace('/', '_')
        safet_file = os.path.join(save_dir, f"{layer_name}.safetensors")
        
        # Get original module properties
        if isinstance(self.module, nn.Linear):
            in_features = self.module.in_features
            out_features = self.module.out_features
            bias = self.module.bias is not None
        elif isinstance(self.module, transformers.Conv1D):
            in_features = self.module.weight.shape[0]
            out_features = self.module.weight.shape[1]
            bias = self.module.bias is not None
        else:
            log.warn(f"Unsupported module type: {type(self.module)}")
            return None
        
        # Determine quant linear class
        quant_linear_candidates = select_quant_linear(
            bits=self.qcfg.bits,
            group_size=self.qcfg.group_size,
            desc_act=self.qcfg.desc_act,
            sym=self.qcfg.sym,
            backend=self.backend,
            format=self.qcfg.format,
            pack=True,
            dynamic=self.qcfg.dynamic,
            device=self.qcfg.device,
            pack_dtype=self.qcfg.pack_dtype,
            multi_select=True,
        )
        
        # Create quantized module
        quantized_module = None
        for quant_linear_cls in quant_linear_candidates:
            try:
                # Validate the quant linear class
                _, err = quant_linear_cls.validate(
                    bits=self.qcfg.bits,
                    group_size=self.qcfg.group_size,
                    desc_act=self.qcfg.desc_act,
                    sym=self.qcfg.sym,
                    pack_dtype=self.qcfg.pack_dtype,
                    in_features=in_features,
                    out_features=out_features,
                    device=self.qcfg.device,
                )
                if err is not None:
                    continue
                
                # Create new quantized module
                quantized_module = quant_linear_cls(
                    bits=self.qcfg.bits,
                    group_size=self.qcfg.group_size,
                    desc_act=self.qcfg.desc_act,
                    sym=self.qcfg.sym,
                    in_features=in_features,
                    out_features=out_features,
                    pack_dtype=self.qcfg.pack_dtype,
                    bias=bias,
                    name=self.name,
                    lm_head_name=self.lm_head_name,
                    backend=self.backend,
                )
                
                # Load quantization data into the module
                quantized_module.qweight = q.to(device='cpu')
                quantized_module.scales = scale.to(device='cpu')
                quantized_module.zeros = zero.to(device='cpu')
                if g_idx is not None:
                    quantized_module.g_idx = g_idx.to(device='cpu')
                
                # Break after successful creation
                break
                
            except Exception as e:
                log.warn(f"Failed to create quantized module {self.name} with {quant_linear_cls.__name__}: {e}")
                continue
        
        if quantized_module is None:
            log.error(f"Failed to create quantized module for {self.name}")
            return None
        
        # Prepare tensors for safetensors format
        # Convert tensors to CPU and numpy format for safetensors
        q_cpu = quantized_module.qweight.cpu().contiguous()
        scale_cpu = quantized_module.scales.cpu().contiguous()
        zero_cpu = quantized_module.zeros.cpu().contiguous()
        g_idx_cpu = quantized_module.g_idx.cpu().contiguous() if quantized_module.g_idx is not None else None
        
        # Create tensors dictionary
        tensors = {}
        
        # Handle different tensor shapes and names based on module type
        if isinstance(self.module, transformers.Conv1D):
            # For Conv1D, we need to transpose back to original format
            q_data = q_cpu.t().numpy()
            tensors[f"{self.name}.qweight"] = q_data
        else:
            # For Linear and Conv2D
            q_data = q_cpu.numpy()
            tensors[f"{self.name}.qweight"] = q_data
            
        scale_data = scale_cpu.numpy()
        tensors[f"{self.name}.scales"] = scale_data
        
        zero_data = zero_cpu.numpy()
        tensors[f"{self.name}.zeros"] = zero_data
        
        if g_idx_cpu is not None:
            g_idx_data = g_idx_cpu.numpy()
            tensors[f"{self.name}.g_idx"] = g_idx_data
        
        # Save to safetensors format
        safetensors.torch.save_file(tensors, safet_file)
        
        # Replace the original module with the quantized one in the model
        parent_name, layer_name = _get_parent_and_layer_name(self)
        if parent_name:
            parent = get_module(self.model, parent_name)
            setattr(parent, layer_name, quantized_module)
        else:
            # If it's a direct child of the model
            layer_name = f"{self.layers_node}.{self.layer_index}"
            parent_name = self.layers_node
            parent = get_module(self.model, parent_name)
            setattr(parent, layer_name, quantized_module)
        
        log.info(f"Memory optimization: Layer {self.name} packed directly to {safet_file}")
        
        # Clean up memory
        if hasattr(self, "H"):
            del self.H
        if hasattr(self, "module_copy"):
            del self.module_copy
        torch.cuda.empty_cache()
        
        return safet_file

    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy
        del self.module
        
        # Clean up temp file if it exists
        if self.qcfg.memory_optimization and hasattr(self, 'temp_file_written') and self.temp_file_written and hasattr(self, 'temp_file_path') and self.temp_file_path:
            try:
                os.remove(self.temp_file_path)
            except:
                pass

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
