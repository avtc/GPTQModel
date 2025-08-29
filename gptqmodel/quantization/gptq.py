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
from ..utils.torch import HAS_CUDA, HAS_XPU, TORCH_GTE_28, DEVICE_0, device_next, torch_compile, torch_sync
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


class GPTQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None):
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
        
        # Set default blocksize from config if not already set
        if not hasattr(self.qcfg, 'block_size') or self.qcfg.block_size is None:
            self.qcfg.block_size = 128

        self.module_copy = None

        self.H = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd input buffer
        self.fwd_inputs_buffered = False
        self.fwd_inputs_buffered_data = []

        # fwd counter
        self.fwd_counter = 0

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
            blocksize=None,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        if blocksize is None:
            blocksize = self.qcfg.block_size
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
            self,
            blocksize=None,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        if blocksize is None:
            blocksize = self.qcfg.block_size
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
            blocksize=None,
    ):
        if blocksize is None:
            blocksize = self.qcfg.block_size
        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        # Temporarily disable torch.compile due to compatibility issues with torch 2.8
        # Will re-enable once the issue is fixed
        if not TORCH_GTE_28:
            self.hessian_inverse = torch_compile(self.hessian_inverse)

        # Use simplified hessian inverse (identity matrix)
        if self.qcfg.mock_hessian_inverse:
            self.hessian_inverse = self._mock_hessian_inverse
            
        # process buffered inputs
        if len(self.fwd_inputs_buffered_data) > 0:
            start_tmp = time.time()

            torch_sync(device=self.module.target_device)

            for inp in self.fwd_inputs_buffered_data:
                self.process_batch(inp)

            # release buffer
            del self.fwd_inputs_buffered_data

            log.debug(f"Completed 1.fwd_inputs_buffered_data for {self.name} in {time.time() - start_tmp:.3f}s")

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

        start_tmp = time.time()

        self.quantizer.find_params(W, weight=True)

        log.debug(f"Completed 2.find_params(W) for {self.name} in {time.time() - start_tmp:.3f}s")

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

        start_loop = time.time()

        if self.qcfg.desc_act:
            start_tmp = time.time()

            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

            log.debug(f"Completed 3.desc_act for {self.name} in {time.time() - start_tmp:.3f}s")

        if hasattr(self.qcfg, "hyb_act") and self.qcfg.hyb_act and not self.qcfg.desc_act:
            from .gar import compose_final_perm, compute_global_perm, compute_local_perms
            start_tmp = time.time()

            local_perms = compute_local_perms(torch.diag(H), self.qcfg.group_size)
            global_perm = compute_global_perm(torch.diag(H), self.qcfg.group_size)
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            W = W[:, final_perm]
            H = H[final_perm][:, final_perm]
            
            log.debug(f"Completed 4.hyb_act for {self.name} in {time.time() - start_tmp:.3f}s")


        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        Hinv, damp = self.hessian_inverse(H)
        
        if self.qcfg.fast_loop:
            # Optimized fast loop implementation with reduced memory allocations
            # Pre-allocate reusable tensors to reduce memory churn
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                # Only allocate error tensors if Hinv is available
                if Hinv is not None:
                    Err1 = torch.zeros_like(W1)
                    Losses1 = torch.zeros_like(W1)

                if Hinv is not None:
                    Hinv1 = Hinv[i1:i2, i1:i2]

                # Handle group quantization parameters efficiently with proper group/block handling
                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        # Check if we can use simple vectorized processing (when block_size is multiple of group_size)
                        can_vectorize_simple = (blocksize % self.qcfg.group_size == 0)
                        
                        if can_vectorize_simple:
                            # Use simple vectorized processing for aligned group/block sizes
                            start_tmp = time.time()
                            
                            # Compute scales/zeros for each group in this block
                            num_groups_in_block = min(blocksize // self.qcfg.group_size,
                                                    (self.columns - i1 + self.qcfg.group_size - 1) // self.qcfg.group_size)
                            
                            block_scales = []
                            block_zeros = []
                            
                            for group_idx in range(num_groups_in_block):
                                global_group_start = i1 + group_idx * self.qcfg.group_size
                                global_group_end = min(global_group_start + self.qcfg.group_size, self.columns)
                                self.quantizer.find_params(W[:, global_group_start:global_group_end], weight=True)
                                block_scales.append(self.quantizer.scale)
                                block_zeros.append(self.quantizer.zero)
                                # Also add to global lists
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                            
                            # Stack group parameters and process all columns in block
                            if len(block_scales) > 0:
                                block_scales = torch.stack(block_scales).view(-1, 1)
                                block_zeros = torch.stack(block_zeros).view(-1, 1)
                                maxq_val = 2 ** self.qcfg.bits - 1
                                
                                # Simple vectorized quantization for the entire block
                                # Create scale and zero tensors that match the block structure
                                # Each group gets its scale/zero applied to its columns in the block
                                # Since block_size is a multiple of group_size, all groups are exactly equal
                                block_scales_full = []
                                block_zeros_full = []
                                
                                for group_idx in range(len(block_scales)):
                                    # Repeat each group's scale/zero for all columns in that group
                                    group_cols = self.qcfg.group_size
                                    
                                    # Repeat the scale/zero for each column in the group
                                    block_scales_full.extend([block_scales[group_idx]] * group_cols)
                                    block_zeros_full.extend([block_zeros[group_idx]] * group_cols)
                                
                                # Convert to tensors and reshape for broadcasting
                                block_scales_full = torch.stack(block_scales_full).view(1, -1)  # Shape: (1, block_size)
                                block_zeros_full = torch.stack(block_zeros_full).view(1, -1)  # Shape: (1, block_size)
                                
                                if self.qcfg.sym:
                                    Q1 = block_scales_full * torch.clamp(
                                        torch.round(W1 / block_scales_full),
                                        -(maxq_val // 2),
                                        maxq_val // 2
                                    )
                                else:
                                    quantized = torch.clamp(
                                        torch.round(W1 / block_scales_full) + block_zeros_full,
                                        0,
                                        maxq_val
                                    )
                                    Q1 = block_scales_full * (quantized - block_zeros_full)
                                
                                log.debug(f"Completed 6.{i1}.Simple vectorized quantization for aligned groups for {self.name} in {time.time() - start_tmp:.3f}s")
                            else:
                                # Fallback to individual processing
                                for i in range(count):
                                    w = W1[:, i]
                                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                                    Q1[:, i] = q
                        else:
                            # Use individual column processing for complex group/block relationships
                            start_tmp = time.time()
                            
                            for i in range(count):
                                col_idx = i1 + i  # Global column index
                                
                                # Only compute scales/zeros for each unique group
                                if i == 0 or (col_idx % self.qcfg.group_size == 0):
                                    group_end = min(col_idx + self.qcfg.group_size, self.columns)
                                    self.quantizer.find_params(W[:, col_idx: group_end], weight=True)
                                    # Also add to the global scale/zero lists for later use
                                    scale.append(self.quantizer.scale)
                                    zero.append(self.quantizer.zero)
                                
                                # Get the column and quantize it
                                w = W1[:, i]
                                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                                Q1[:, i] = q
                            
                            log.debug(f"Completed 6.{i1}.Individual quantization for complex groups for {self.name} in {time.time() - start_tmp:.3f}s")
                    else:
                        # Static groups - optimized individual processing
                        # Process columns with optimized loop while maintaining group logic
                        for i in range(count):
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]
                            
                            # Get the column and quantize it
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q
                else:
                    # No grouping - optimized individual quantization
                    # Process all columns with optimized loop for better performance and cache utilization
                    for i in range(count):
                        w = W1[:, i]
                        q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                        Q1[:, i] = q

                # Optimized error computation - eliminates inner loop and unnecessary W1 updates
                if Hinv is not None:
                    start_tmp = time.time()

                    # Precompute diagonal values and inverse for vectorized operations
                    diag_vals = torch.diag(Hinv1)  # Shape: (count,)
                    inv_diag = 1.0 / diag_vals  # Shape: (count,)
                    
                    # Vectorized computation of all errors and losses
                    W_cols = W1.T  # Shape: (count, rows) for column-wise operations
                    Q_cols = Q1.T  # Shape: (count, rows)
                    
                    # Vectorized error computation for all columns
                    diff = W_cols - Q_cols  # Shape: (count, rows)
                    errors = diff * inv_diag.unsqueeze(1)  # Shape: (count, rows)
                    
                    # Vectorized loss computation - fix dimension mismatch
                    # diff: (count, rows), errors: (count, rows)
                    # Compute element-wise product and sum along rows for each column
                    Losses1 = (diff * errors).T  # Shape: (rows, count)
                    
                    # Store all errors - no need to update W1 as it's not used after this block
                    Err1 = errors.T  # Shape: (rows, count) back to original shape

                    log.debug(f"Completed 9.{i1}.Optimized error computation for {self.name} in {time.time() - start_tmp:.3f}s")


                start_tmp = time.time()

                Q[:, i1:i2] = Q1
                if Hinv is not None:
                    Losses[:, i1:i2] = Losses1 / 2
                    # Use in-place operation for final update
                    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                log.debug(f"Completed 10.{i1}.Final update for {self.name} in {time.time() - start_tmp:.3f}s")

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
                            # Only compute scales/zeros when we're at the start of a new group
                            if (i1 + i) % self.qcfg.group_size == 0:
                                group_end = min(i1 + i + self.qcfg.group_size, self.columns)
                                self.quantizer.find_params(W[:, (i1 + i):group_end], weight=True)

                            # Add scale/zero to global list when we complete a group
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

        log.debug(f"Completed Loop for {self.name} in {time.time() - start_loop:.3f}s")

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
        if Q.device != DEVICE_0:
            try:
                Q = Q.to(device=DEVICE_0)
            except Exception as e:
                log.warn(f"Failed to move Q from {Q.device.type}:{Q.device.index} to {DEVICE_0.type}:{DEVICE_0.index}, {e}")
                if Q.device != DEVICE_0:
                    try:
                        Q = Q.to(device=DEVICE_0)
                    except Exception as e2:
                        log.error(f"Failed to move Q from {Q.device.type}:{Q.device.index} to {DEVICE_0.type}:{DEVICE_0.index}, {e2} (second attempt)")
                        raise

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

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy
        del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
