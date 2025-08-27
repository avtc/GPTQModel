#!/usr/bin/env python3
"""
Test script to verify that the optimized fast_loop produces correct results
compared to the original implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization import QuantizeConfig

def create_test_model():
    """Create a simple test model for verification"""
    return nn.Linear(64, 32, bias=False)

def create_test_data():
    """Create test data for quantization"""
    batch_size = 8
    seq_len = 16
    input_dim = 64
    
    # Create random input data
    torch.manual_seed(42)  # For reproducibility
    inputs = torch.randn(batch_size, seq_len, input_dim)
    
    return inputs

def test_fast_loop_correctness():
    """Test that fast_loop produces correct results compared to original"""
    print("Testing fast_loop correctness...")
    
    # Create test model and data
    model = create_test_model()
    test_data = create_test_data()
    
    # Create quantization config
    qcfg = QuantizeConfig(bits=4, group_size=16, desc_act=False, sym=True, fast_loop=True)
    
    # Create two identical GPTQ instances
    gptq_fast = GPTQ(model, qcfg)
    gptq_original = GPTQ(model, qcfg)
    
    # Reset qcfg to disable fast_loop for original
    gptq_original.qcfg.fast_loop = False
    
    # Add the same batch to both instances
    for _ in range(5):  # Add multiple batches for better Hessian estimation
        gptq_fast.add_batch(test_data, model(test_data))
        gptq_original.add_batch(test_data, model(test_data))
    
    # Quantize using both implementations
    try:
        Q_fast, scale_fast, zero_fast, g_idx_fast, duration_fast, avg_loss_fast, damp_fast, nsamples_fast = gptq_fast.quantize()
        Q_original, scale_original, zero_original, g_idx_original, duration_original, avg_loss_original, damp_original, nsamples_original = gptq_original.quantize()
        
        # Compare results
        print(f"Fast loop - Loss: {avg_loss_fast:.6f}, Duration: {duration_fast:.3f}s")
        print(f"Original - Loss: {avg_loss_original:.6f}, Duration: {duration_original:.3f}s")
        
        # Check if quantized weights are close
        weight_diff = torch.max(torch.abs(Q_fast - Q_original)).item()
        print(f"Max weight difference: {weight_diff:.2e}")
        
        # Check if scales and zeros are close
        scale_diff = torch.max(torch.abs(scale_fast - scale_original)).item() if scale_fast.shape == scale_original.shape else float('inf')
        zero_diff = torch.max(torch.abs(zero_fast - zero_original)).item() if zero_fast.shape == zero_original.shape else float('inf')
        print(f"Max scale difference: {scale_diff:.2e}")
        print(f"Max zero difference: {zero_diff:.2e}")
        
        # Check if loss difference is acceptable (within 1%)
        loss_diff = abs(avg_loss_fast - avg_loss_original) / avg_loss_original
        print(f"Relative loss difference: {loss_diff:.2%}")
        
        # Success criteria
        success = (weight_diff < 1e-5 and 
                  loss_diff < 0.01 and 
                  scale_diff < 1e-5 and 
                  zero_diff < 1e-5)
        
        if success:
            print("✓ Test PASSED: Fast loop produces correct results!")
        else:
            print("✗ Test FAILED: Fast loop results differ significantly from original!")
            
        return success
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def test_error_computation_correctness():
    """Test that error computation is mathematically correct"""
    print("\nTesting error computation correctness...")
    
    # Create simple test case
    torch.manual_seed(42)
    W1 = torch.randn(32, 8)  # Test weight matrix
    Q1 = torch.randn(32, 8)  # Quantized weights
    Hinv1 = torch.randn(8, 8)
    Hinv1 = Hinv1 @ Hinv1.T  # Make symmetric positive definite
    Hinv1 += torch.eye(8) * 2  # Ensure positive definite
    
    # Test original error computation logic
    Losses1_original = torch.zeros_like(W1)
    Err1_original = torch.zeros_like(W1)
    
    for i in range(8):
        w = W1[:, i]
        q = Q1[:, i]
        d = Hinv1[i, i]
        
        Losses1_original[:, i] = (w - q) ** 2 / d**2
        err1 = (w - q) / d
        W1_temp = W1.clone()
        W1_temp[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        Err1_original[:, i] = err1
    
    # Test optimized error computation
    differences = W1 - Q1
    diagonal_elements = Hinv1.diagonal()
    
    Losses1_optimized = (differences ** 2) / (diagonal_elements.unsqueeze(1) ** 2)
    errors = differences / diagonal_elements.unsqueeze(1)
    Err1_optimized = errors.clone()
    
    W1_test = W1.clone()
    for i in range(8):
        W1_test[:, i:] -= errors[:, i:i+1] @ Hinv1[i:i+1, i:]
    
    # Compare results
    loss_diff = torch.max(torch.abs(Losses1_original - Losses1_optimized)).item()
    err_diff = torch.max(torch.abs(Err1_original - Err1_optimized)).item()
    weight_diff = torch.max(torch.abs(W1_temp - W1_test)).item()
    
    print(f"Loss computation difference: {loss_diff:.2e}")
    print(f"Error computation difference: {err_diff:.2e}")
    print(f"Weight update difference: {weight_diff:.2e}")
    
    success = (loss_diff < 1e-10 and err_diff < 1e-10 and weight_diff < 1e-10)
    
    if success:
        print("✓ Error computation test PASSED!")
    else:
        print("✗ Error computation test FAILED!")
        
    return success

if __name__ == "__main__":
    print("Running fast_loop correctness tests...\n")
    
    test1_result = test_fast_loop_correctness()
    test2_result = test_error_computation_correctness()
    
    if test1_result and test2_result:
        print("\n🎉 All tests PASSED! The optimized fast_loop implementation is correct.")
    else:
        print("\n❌ Some tests FAILED. Please review the implementation.")