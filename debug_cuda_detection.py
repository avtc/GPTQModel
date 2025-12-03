#!/usr/bin/env python3

import os
import torch

print("=== PyTorch CUDA Detection ===")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

print("\n=== GPTQModel Device Detection ===")

# Import after torch detection
from gptqmodel.utils.torch import HAS_CUDA, ALL_DEVICES
from gptqmodel import DEVICE_THREAD_POOL

print(f"HAS_CUDA: {HAS_CUDA}")
print(f"ALL_DEVICES: {ALL_DEVICES}")

# Check DeviceThreadPool state
print(f"\n=== DeviceThreadPool State ===")
print(f"_empty_cache_every_n: {DEVICE_THREAD_POOL._empty_cache_every_n}")
print(f"_devices_by_key: {DEVICE_THREAD_POOL._devices_by_key}")
print(f"_ordered_keys: {DEVICE_THREAD_POOL._ordered_keys}")

# Check if any accelerators are detected
accelerator_devices = [
    key for key in DEVICE_THREAD_POOL._ordered_keys
    if DEVICE_THREAD_POOL._devices_by_key[key].type in ("cuda", "xpu", "mps")
]

print(f"Accelerator devices found: {accelerator_devices}")

# Check the exact condition that determines if DP-Janitor starts
should_start_janitor = (
    DEVICE_THREAD_POOL._empty_cache_every_n > 0 and any(
        DEVICE_THREAD_POOL._devices_by_key[k].type in ("cuda", "xpu", "mps") 
        for k in DEVICE_THREAD_POOL._ordered_keys
    )
)

print(f"Should start DP-Janitor: {should_start_janitor}")
print(f"  _empty_cache_every_n > 0: {DEVICE_THREAD_POOL._empty_cache_every_n > 0}")
print(f"  Any accelerator devices: {any(DEVICE_THREAD_POOL._devices_by_key[k].type in ('cuda', 'xpu', 'mps') for k in DEVICE_THREAD_POOL._ordered_keys)}")

print("\n=== Environment Variables ===")
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

# Try to manually create a DeviceThreadPool with CUDA
print("\n=== Manual DeviceThreadPool Test ===")
from gptqmodel.utils.threadx import DeviceThreadPool

manual_pool = DeviceThreadPool(
    devices=["cuda:0"] if torch.cuda.is_available() else ["cpu"],
    empty_cache_every_n=10
)

print(f"Manual pool _devices_by_key: {manual_pool._devices_by_key}")
print(f"Manual pool should start janitor: {manual_pool._empty_cache_every_n > 0 and any(manual_pool._devices_by_key[k].type in ('cuda', 'xpu', 'mps') for k in manual_pool._ordered_keys)}")