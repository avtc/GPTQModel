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
        groups = None

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

        perm = None
        invperm = None
        
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
            Q, Losses, W, scale, zero, now_idx = self.fast_loop2(W, Q, Losses, Hinv, blocksize, perm, invperm, groups, scale, zero, now_idx)
            # # Optimized fast loop implementation with reduced memory allocations
            # # Pre-allocate reusable tensors to reduce memory churn
            # for i1 in range(0, self.columns, blocksize):
            #     i2 = min(i1 + blocksize, self.columns)
            #     count = i2 - i1

            #     W1 = W[:, i1:i2].clone()
            #     Q1 = torch.zeros_like(W1)
            #     # Only allocate error tensors if Hinv is available
            #     if Hinv is not None:
            #         Err1 = torch.zeros_like(W1)
            #         Losses1 = torch.zeros_like(W1)

            #     if Hinv is not None:
            #         Hinv1 = Hinv[i1:i2, i1:i2]

            #     # Handle group quantization parameters efficiently with proper group/block handling
            #     if self.qcfg.group_size != -1:
            #         if not self.qcfg.static_groups:
            #             # NEW: Always iterate over groups within each block, regardless of alignment
            #             start_tmp = time.time()
                        
            #             # Find all groups that intersect with this block
            #             block_start_group = i1 // self.qcfg.group_size
            #             block_end_group = (i2 - 1) // self.qcfg.group_size
            #             groups_in_block = range(block_start_group, block_end_group + 1)
                        
            #             # Track which groups we've already processed to avoid recomputation
            #             processed_groups = set()
                        
            #             # Create mapping from local column indices to group scales/zeros
            #             col_to_scale = []
            #             col_to_zero = []
                        
            #             for i in range(count):
            #                 col_idx = i1 + i  # Global column index
            #                 group_idx = col_idx // self.qcfg.group_size
                            
            #                 if group_idx not in processed_groups:
            #                     # This is the first time we're processing this group
            #                     group_start = group_idx * self.qcfg.group_size
            #                     group_end = min(group_start + self.qcfg.group_size, self.columns)
                                
            #                     self.quantizer.find_params(W[:, group_start:group_end], weight=True)
                                
            #                     # Store scale and zero for this group
            #                     col_to_scale.append(self.quantizer.scale)
            #                     col_to_zero.append(self.quantizer.zero)
                                
            #                     # Add to global lists
            #                     scale.append(self.quantizer.scale)
            #                     zero.append(self.quantizer.zero)
                                
            #                     processed_groups.add(group_idx)
            #                 else:
            #                     # We've already processed this group, use the stored scale/zero
            #                     group_pos_in_block = list(groups_in_block).index(group_idx)
            #                     col_to_scale.append(col_to_scale[group_pos_in_block])
            #                     col_to_zero.append(col_to_zero[group_pos_in_block])
                        
            #             # Convert to tensors for vectorized operations
            #             if len(col_to_scale) > 0:
            #                 col_scales = torch.stack(col_to_scale).view(-1, 1)
            #                 col_zeros = torch.stack(col_to_zero).view(-1, 1)
            #                 maxq_val = 2 ** self.qcfg.bits - 1
                            
            #                 # Vectorized quantization for all columns in the block
            #                 if self.qcfg.sym:
            #                     Q1 = col_scales * torch.clamp(
            #                         torch.round(W1 / col_scales),
            #                         -(maxq_val // 2),
            #                         maxq_val // 2
            #                     )
            #                 else:
            #                     quantized = torch.clamp(
            #                         torch.round(W1 / col_scales) + col_zeros,
            #                         0,
            #                         maxq_val
            #                     )
            #                     Q1 = col_scales * (quantized - col_zeros)
            #             else:
            #                 # Fallback to individual processing (shouldn't happen with proper logic)
            #                 for i in range(count):
            #                     w = W1[:, i]
            #                     q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
            #                     Q1[:, i] = q
                        
            #             log.debug(f"Completed 6.{i1}.Group-aware vectorized quantization for {self.name} in {time.time() - start_tmp:.3f}s")
            #         else:
            #             # Static groups - optimized individual processing
            #             # Process columns with optimized loop while maintaining group logic
            #             for i in range(count):
            #                 idx = i1 + i
            #                 if self.qcfg.desc_act:
            #                     idx = perm[idx]
            #                 self.quantizer = groups[idx // self.qcfg.group_size]
                            
            #                 # Get the column and quantize it
            #                 w = W1[:, i]
            #                 q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
            #                 Q1[:, i] = q
            #     else:
            #         # No grouping - optimized individual quantization
            #         # Process all columns with optimized loop for better performance and cache utilization
            #         for i in range(count):
            #             w = W1[:, i]
            #             q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
            #             Q1[:, i] = q

            #     # Optimized error computation - eliminates inner loop and unnecessary W1 updates
            #     if Hinv is not None:
            #         start_tmp = time.time()

            #         # Precompute diagonal values and inverse for vectorized operations
            #         diag_vals = torch.diag(Hinv1)  # Shape: (count,)
            #         inv_diag = 1.0 / diag_vals  # Shape: (count,)
                    
            #         # Vectorized computation of all errors and losses
            #         W_cols = W1.T  # Shape: (count, rows) for column-wise operations
            #         Q_cols = Q1.T  # Shape: (count, rows)
                    
            #         # Vectorized error computation for all columns
            #         diff = W_cols - Q_cols  # Shape: (count, rows)
            #         errors = diff * inv_diag.unsqueeze(1)  # Shape: (count, rows)
                    
            #         # Vectorized loss computation - fix dimension mismatch
            #         # diff: (count, rows), errors: (count, rows)
            #         # Compute element-wise product and sum along rows for each column
            #         Losses1 = (diff * errors).T  # Shape: (rows, count)
                    
            #         # Store all errors - no need to update W1 as it's not used after this block
            #         Err1 = errors.T  # Shape: (rows, count) back to original shape

            #         log.debug(f"Completed 9.{i1}.Optimized error computation for {self.name} in {time.time() - start_tmp:.3f}s")


            #     start_tmp = time.time()

            #     Q[:, i1:i2] = Q1
            #     if Hinv is not None:
            #         Losses[:, i1:i2] = Losses1 / 2
            #         # Use in-place operation for final update
            #         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            #     log.debug(f"Completed 10.{i1}.Final update for {self.name} in {time.time() - start_tmp:.3f}s")

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

    def fast_loop2(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        Losses: torch.Tensor,
        Hinv: torch.Tensor,
        blocksize: int,
        perm: Optional[torch.Tensor] = None,
        invperm: Optional[torch.Tensor] = None,
        groups: Optional[list] = None,
        scale: Optional[list] = None,
        zero: Optional[list] = None,
        now_idx: int = 1,
    ):
        """
        Optimized fast loop implementation with improved memory efficiency and vectorization.
        Based on the original heavy loop but with significant performance optimizations.
        
        Args:
            W: Weight matrix to quantize
            Q: Output quantized weight matrix
            Losses: Output loss matrix
            Hinv: Hessian inverse matrix
            blocksize: Size of processing blocks
            perm: Optional permutation for desc_act
            invperm: Optional inverse permutation for desc_act
            groups: Optional list of quantizers for static groups
            scale: List to store scale values
            zero: List to store zero values
            now_idx: Current index for group tracking
            
        Returns:
            Updated Q, Losses, W, scale, zero, now_idx
        """
        # Initialize default lists if not provided
        if scale is None:
            scale = []
        if zero is None:
            zero = []
            
        if Hinv is not None:
            Hinv = Hinv.to(device=W.device)
        
        # Pre-compute diagonal values for error computation
        if Hinv is not None:
            diag_vals = torch.diag(Hinv)
            inv_diag = 1.0 / diag_vals
        
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            
            # Clone only the current block to reduce memory usage
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            
            if Hinv is not None:
                Hinv1 = Hinv[i1:i2, i1:i2]
            
            # Optimized group quantization with pre-computation
            if self.qcfg.group_size != -1:
                if not self.qcfg.static_groups:
                    # Pre-compute group mappings for the entire block
                    block_start_group = i1 // self.qcfg.group_size
                    block_end_group = (i2 - 1) // self.qcfg.group_size
                    
                    # Cache group quantization parameters to avoid repeated calls
                    group_cache = {}
                    col_to_group_scale = []
                    col_to_group_zero = []
                    
                    log.debug(f"fast_loop2 DEBUG - Processing block from {i1} to {i2}, count={count}")
                    log.debug(f"fast_loop2 DEBUG - Total columns: {self.columns}, group_size: {self.qcfg.group_size}")
                    
                    for i in range(count):
                        col_idx = i1 + i
                        group_idx = col_idx // self.qcfg.group_size
                        
                        log.debug(f"fast_loop2 DEBUG - Column {col_idx}, group {group_idx}")
                        
                        if group_idx not in group_cache:
                            # First time processing this group - compute parameters
                            group_start = group_idx * self.qcfg.group_size
                            group_end = min(group_start + self.qcfg.group_size, self.columns)
                            
                            log.debug(f"fast_loop2 DEBUG - New group {group_idx}: range {group_start} to {group_end}")
                            
                            self.quantizer.find_params(W[:, group_start:group_end], weight=True)
                            
                            # DEBUG: Log what the quantizer returns
                            log.debug(f"fast_loop2 DEBUG - quantizer.scale shape: {self.quantizer.scale.shape if hasattr(self.quantizer.scale, 'shape') else 'no shape'}")
                            log.debug(f"fast_loop2 DEBUG - quantizer.zero shape: {self.quantizer.zero.shape if hasattr(self.quantizer.zero, 'shape') else 'no shape'}")
                            
                            # Store parameters for this group - keep the original tensor shapes
                            group_cache[group_idx] = {
                                'scale': self.quantizer.scale,
                                'zero': self.quantizer.zero
                            }
                            
                            # Add to global lists only once per group
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                            
                            # Create scale/zero for this group (only one per group, not per column)
                            group_cols = min(group_end - group_start, count - i if group_idx == block_end_group else self.qcfg.group_size)
                            log.debug(f"fast_loop2 DEBUG - Adding 1 scale/zero for group {group_idx} that covers {group_cols} columns")
                            
                            # Only append one scale/zero for the entire group - squeeze to remove the last dimension
                            col_to_group_scale.append(self.quantizer.scale.squeeze(-1))  # Shape: [64]
                            col_to_group_zero.append(self.quantizer.zero.squeeze(-1))    # Shape: [64]
                        else:
                            # Use cached parameters for this group
                            cached = group_cache[group_idx]
                            group_cols = min(self.qcfg.group_size, count - i if group_idx == block_end_group else self.qcfg.group_size)
                            
                            log.debug(f"fast_loop2 DEBUG - Using cached group {group_idx} for {group_cols} columns")
                            
                            # Only append one cached scale/zero for the entire group - squeeze to remove the last dimension
                            col_to_group_scale.append(cached['scale'].squeeze(-1))  # Shape: [64]
                            col_to_group_zero.append(cached['zero'].squeeze(-1))    # Shape: [64]
                    
                    log.debug(f"fast_loop2 DEBUG - Total scales collected: {len(col_to_group_scale)}")
                    log.debug(f"fast_loop2 DEBUG - Total zeros collected: {len(col_to_group_zero)}")
                    
                    # Vectorized quantization for all columns in the block
                    if len(col_to_group_scale) > 0:
                        start_tmp = time.time()
                        
                        # Stack all group parameters for vectorized operations
                        block_scales = torch.stack(col_to_group_scale)  # Shape: (count, 1024, 1)
                        block_zeros = torch.stack(col_to_group_zero)    # Shape: (count, 1024, 1)
                        maxq_val = 2 ** self.qcfg.bits - 1
                        
                        # DEBUG: Log tensor shapes for debugging
                        log.debug(f"fast_loop2 DEBUG - W1 shape: {W1.shape}")
                        log.debug(f"fast_loop2 DEBUG - block_scales shape before view: {block_scales.shape}")
                        log.debug(f"fast_loop2 DEBUG - block_zeros shape before view: {block_zeros.shape}")
                        log.debug(f"fast_loop2 DEBUG - count: {count}, rows: {W1.shape[0]}")
                        
                        # We need to reshape from (count, rows) to (rows, count) for proper broadcasting with W1
                        # W1 has shape (rows, count), so we need scales/zeros with shape (rows, count)
                        block_scales = block_scales.T  # Shape: (64, 8)
                        block_zeros = block_zeros.T    # Shape: (64, 8)
                        
                        log.debug(f"fast_loop2 DEBUG - block_scales shape after view: {block_scales.shape}")
                        log.debug(f"fast_loop2 DEBUG - block_zeros shape after view: {block_zeros.shape}")
                        
                        # Vectorized quantization following the working pattern
                        if self.qcfg.sym:
                            # Symmetric quantization: Q = scale * clamp(round(x/scale), -maxq/2, maxq/2)
                            Q1 = block_scales * torch.clamp(
                                torch.round(W1 / block_scales),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            log.debug(f"fast_loop2 DEBUG - About to perform quantized calculation")
                            # Asymmetric quantization: Q = scale * (clamp(round(x/scale) + zero, 0, maxq) - zero)
                            quantized = torch.clamp(
                                torch.round(W1 / block_scales) + block_zeros,
                                0,
                                maxq_val
                            )
                            Q1 = block_scales * (quantized - block_zeros)
                            log.debug(f"fast_loop2 DEBUG - Quantized calculation completed")
                        
                        log.debug(f"fast_loop2 DEBUG - Completed vectorized quantization for {self.name} in {time.time() - start_tmp:.3f}s")
                else:
                    # Static groups - optimized processing
                    for i in range(count):
                        idx = i1 + i
                        if self.qcfg.desc_act and perm is not None:
                            idx = perm[idx]
                        self.quantizer = groups[idx // self.qcfg.group_size]
                        
                        # Process column with optimized quantization
                        w = W1[:, i]
                        q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                        Q1[:, i] = q
            else:
                # No grouping - optimized individual processing
                for i in range(count):
                    w = W1[:, i]
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
            
            # Optimized error computation with vectorized operations
            if Hinv is not None:
                # Vectorized computation of all errors and losses
                W_cols = W1.T  # Transpose for column-wise operations
                Q_cols = Q1.T
                
                # Compute differences and errors for all columns
                diff = W_cols - Q_cols
                
                # Use pre-computed inverse diagonal values
                if i1 < len(inv_diag):
                    block_inv_diag = inv_diag[i1:i2].view(-1, 1)
                    errors = diff * block_inv_diag
                    
                    # Vectorized loss computation
                    Losses1 = (diff * errors).T
                    
                    # Store errors for final update
                    Err1 = errors.T
                    
                    # Update remaining weights in one matrix operation
                    if i2 < self.columns:
                        # Hinv1 has shape (blocksize, blocksize), we need to use all of it
                        W[:, i2:] -= Err1.matmul(Hinv1)
            
            # Store results
            Q[:, i1:i2] = Q1
            if Hinv is not None:
                Losses[:, i1:i2] = Losses1 / 2
        
        return Q, Losses, W, scale, zero, now_idx

    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy
        del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
