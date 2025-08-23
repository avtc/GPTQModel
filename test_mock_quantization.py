#!/usr/bin/env python3

"""
Test script to verify mock quantization functionality
"""

import torch
import torch.nn as nn
import numpy as np
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.models.base import BaseGPTQModel
from transformers import AutoModelForCausalLM

def test_mock_quantization():
    """Test that mock quantization works correctly"""
    
    # Create a simple test model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Create test data
    test_input = torch.randn(32, 128)
    
    # Test 1: Normal quantization (should be slower)
    print("Testing normal quantization...")
    qcfg_normal = QuantizeConfig(bits=4, group_size=128, desc_act=True)
    gptq_normal = GPTQ(model, qcfg_normal)
    
    # Add some batches to simulate calibration data
    for _ in range(5):
        output = model(test_input)
        gptq_normal.add_batch(test_input, output)
    
    start_time = time.time()
    result_normal = gptq_normal.quantize()
    normal_time = time.time() - start_time
    
    print(f"Normal quantization took {normal_time:.2f} seconds")
    print(f"Normal quantization result shape: {result_normal[0].shape}")
    
    # Test 2: Mock quantization (should be much faster)
    print("\nTesting mock quantization...")
    qcfg_mock = QuantizeConfig(bits=4, group_size=128, desc_act=True, mock_quantization=True)
    gptq_mock = GPTQ(model, qcfg_mock)
    
    # Add same test data
    for _ in range(5):
        output = model(test_input)
        gptq_mock.add_batch(test_input, output)
    
    start_time = time.time()
    result_mock = gptq_mock.quantize()
    mock_time = time.time() - start_time
    
    print(f"Mock quantization took {mock_time:.2f} seconds")
    print(f"Mock quantization result shape: {result_mock[0].shape}")
    print(f"Speedup: {normal_time/mock_time:.1f}x faster")
    
    # Verify results are different (mock should be different from normal)
    assert not torch.allclose(result_normal[0], result_mock[0]), "Mock and normal results should be different"
    
    # Verify mock uses identity matrix for hessian
    H_test = torch.eye(10)
    Hinv_mock, damp_mock = gptq_mock._mock_hessian_inverse(H_test)
    assert torch.allclose(Hinv_mock, torch.eye(10)), "Mock hessian should be identity matrix"
    
    print("\n✅ Mock quantization test passed!")
    print(f"✅ Mock quantization is {normal_time/mock_time:.1f}x faster than normal quantization")

if __name__ == "__main__":
    import time
    test_mock_quantization()