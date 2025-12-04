"""Unit tests for batch distribution across GPUs.

Tests verify that the round-robin distribution creates better token
balance compared to contiguous segment distribution when samples are
pre-sorted by length in descending order.
"""

import pytest


def test_round_robin_distribution_correctness():
    """Verify batches are distributed round-robin across devices."""
    total_batches = 7
    num_devices = 3
    devices = [f"cuda:{i}" for i in range(num_devices)]
    
    # Simulate the round-robin distribution logic from module_looper.py
    device_segments = {device: [] for device in devices}
    
    for batch_idx in range(total_batches):
        device_idx = batch_idx % num_devices
        device = devices[device_idx]
        device_segments[device].append(batch_idx)
    
    # Verify distribution
    # With 7 batches and 3 devices: GPU0=[0,3,6], GPU1=[1,4], GPU2=[2,5]
    assert device_segments["cuda:0"] == [0, 3, 6]
    assert device_segments["cuda:1"] == [1, 4]
    assert device_segments["cuda:2"] == [2, 5]
    
    # Verify all batches are assigned
    all_batches = []
    for batches in device_segments.values():
        all_batches.extend(batches)
    assert sorted(all_batches) == list(range(total_batches))


def test_round_robin_with_two_devices():
    """Test round-robin with 2 GPUs."""
    total_batches = 6
    num_devices = 2
    devices = [f"cuda:{i}" for i in range(num_devices)]
    
    device_segments = {device: [] for device in devices}
    
    for batch_idx in range(total_batches):
        device_idx = batch_idx % num_devices
        device = devices[device_idx]
        device_segments[device].append(batch_idx)
    
    # GPU0 should get even indices, GPU1 odd indices
    assert device_segments["cuda:0"] == [0, 2, 4]
    assert device_segments["cuda:1"] == [1, 3, 5]


def test_token_balance_improvement_with_sorted_samples():
    """Verify round-robin improves balance vs contiguous segments when sorted."""
    # Simulate batches sorted in descending order by token count
    # (as done by calibration.py with calibration_dataset_sort="desc")
    batch_token_counts = [1000, 900, 800, 700, 100, 50]
    total_batches = len(batch_token_counts)
    num_devices = 2
    
    # OLD APPROACH: Contiguous segments
    # GPU 0: batches [0, 1, 2] = 1000 + 900 + 800 = 2700
    # GPU 1: batches [3, 4, 5] = 700 + 100 + 50 = 850
    mid = total_batches // 2
    old_gpu0_tokens = sum(batch_token_counts[:mid])
    old_gpu1_tokens = sum(batch_token_counts[mid:])
    old_max_tokens = max(old_gpu0_tokens, old_gpu1_tokens)
    old_min_tokens = min(old_gpu0_tokens, old_gpu1_tokens)
    old_imbalance_ratio = old_max_tokens / old_min_tokens if old_min_tokens > 0 else float('inf')
    
    # NEW APPROACH: Round-robin
    # GPU 0: batches [0, 2, 4] = 1000 + 800 + 100 = 1900
    # GPU 1: batches [1, 3, 5] = 900 + 700 + 50 = 1650
    new_gpu_loads = [0] * num_devices
    for batch_idx in range(total_batches):
        device_idx = batch_idx % num_devices
        new_gpu_loads[device_idx] += batch_token_counts[batch_idx]
    
    new_max_tokens = max(new_gpu_loads)
    new_min_tokens = min(new_gpu_loads)
    new_imbalance_ratio = new_max_tokens / new_min_tokens if new_min_tokens > 0 else float('inf')
    
    # Verify specific token counts
    assert new_gpu_loads[0] == 1900  # GPU 0: 1000 + 800 + 100
    assert new_gpu_loads[1] == 1650  # GPU 1: 900 + 700 + 50
    
    # Verify round-robin is significantly better
    assert new_imbalance_ratio < old_imbalance_ratio
    
    # Verify new approach has acceptable balance (within 20%)
    imbalance_percentage = (new_max_tokens - new_min_tokens) / new_max_tokens
    assert imbalance_percentage < 0.2, \
        f"Imbalance {imbalance_percentage:.1%} exceeds 20% threshold"
    
    # Old approach should be much worse (demonstrate the problem we're solving)
    old_imbalance_percentage = (old_max_tokens - old_min_tokens) / old_max_tokens
    assert old_imbalance_percentage > 0.5, \
        f"Old approach should have >50% imbalance, got {old_imbalance_percentage:.1%}"


def test_token_balance_with_three_devices():
    """Test token balance with 3 GPUs and sorted samples."""
    # Sorted descending
    batch_token_counts = [2000, 1500, 1000, 800, 600, 400, 200, 100, 50]
    total_batches = len(batch_token_counts)
    num_devices = 3
    
    # Round-robin distribution
    gpu_loads = [0] * num_devices
    for batch_idx in range(total_batches):
        device_idx = batch_idx % num_devices
        gpu_loads[device_idx] += batch_token_counts[batch_idx]
    
    # GPU 0: batches [0, 3, 6] = 2000 + 800 + 200 = 3000
    # GPU 1: batches [1, 4, 7] = 1500 + 600 + 100 = 2200
    # GPU 2: batches [2, 5, 8] = 1000 + 400 + 50 = 1450
    assert gpu_loads[0] == 3000
    assert gpu_loads[1] == 2200
    assert gpu_loads[2] == 1450
    
    # Verify reasonable balance
    max_load = max(gpu_loads)
    min_load = min(gpu_loads)
    imbalance_ratio = max_load / min_load
    
    # With sorted descending data, round-robin should keep ratio < 2.5
    assert imbalance_ratio < 2.5, f"Imbalance ratio {imbalance_ratio:.2f} too high"


def test_edge_case_single_device():
    """Test that single device gets all batches."""
    total_batches = 5
    num_devices = 1
    devices = ["cuda:0"]
    
    device_segments = {device: [] for device in devices}
    
    for batch_idx in range(total_batches):
        device_idx = batch_idx % num_devices
        device = devices[device_idx]
        device_segments[device].append(batch_idx)
    
    # Single device should get all batches
    assert device_segments["cuda:0"] == [0, 1, 2, 3, 4]


def test_edge_case_more_devices_than_batches():
    """Test when there are more devices than batches."""
    total_batches = 2
    num_devices = 5
    devices = [f"cuda:{i}" for i in range(num_devices)]
    
    device_segments = {device: [] for device in devices}
    
    for batch_idx in range(total_batches):
        device_idx = batch_idx % num_devices
        device = devices[device_idx]
        device_segments[device].append(batch_idx)
    
    # Only first 2 devices get batches
    assert device_segments["cuda:0"] == [0]
    assert device_segments["cuda:1"] == [1]
    assert device_segments["cuda:2"] == []
    assert device_segments["cuda:3"] == []
    assert device_segments["cuda:4"] == []


def test_edge_case_empty_batches():
    """Test with zero batches."""
    total_batches = 0
    num_devices = 3
    devices = [f"cuda:{i}" for i in range(num_devices)]
    
    device_segments = {device: [] for device in devices}
    
    for batch_idx in range(total_batches):
        device_idx = batch_idx % num_devices
        device = devices[device_idx]
        device_segments[device].append(batch_idx)
    
    # All devices should be empty
    for device in devices:
        assert device_segments[device] == []
