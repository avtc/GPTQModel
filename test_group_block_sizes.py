#!/usr/bin/env python3
"""
Test script to verify that GPTQ quantization works with different group_size and blocksize combinations.
"""

from gptqmodel.quantization.gptq import GPTQ
import torch
import torch.nn as nn
import numpy as np
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_test_model():
    """Create a simple test model for quantization."""
    # Create a simple linear layer for testing
    layer = nn.Linear(256, 512)
    # Initialize with some test data
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    return layer

def test_quantization_configs():
    """Test various group_size and blocksize combinations."""
    
    test_cases = [
        # (group_size, blocksize, description)
        (64, 128, "group_size=64, blocksize=128 (aligned)"),
        (32, 128, "group_size=32, blocksize=128 (aligned)"),
        (16, 128, "group_size=16, blocksize=128 (aligned)"),
        (128, 128, "group_size=128, blocksize=128 (aligned)"),
        (64, 96, "group_size=64, blocksize=96 (unaligned)"),
        (32, 96, "group_size=32, blocksize=96 (unaligned)"),
        (16, 96, "group_size=16, blocksize=96 (unaligned)"),
        (128, 96, "group_size=128, blocksize=96 (unaligned)"),
        (64, 256, "group_size=64, blocksize=256 (unaligned)"),
    ]
    
    print("Testing GPTQ quantization with different group_size and blocksize combinations...")
    print("=" * 80)
    
    for group_size, blocksize, description in test_cases:
        print(f"\nTesting: {description}")
        
        try:
            # Create test model
            model = create_test_model()
            
            # Create quantization config
            qcfg = QuantizeConfig(
                bits=4,
                group_size=group_size,
                block_size=blocksize,
                sym=True,
                desc_act=False,  # Disable activation ordering for simpler testing
                fast_loop=True   # Use our optimized implementation
            )
            
            # Create GPTQ wrapper
            gptq = GPTQ(model, qcfg)
            
            # Create mock Hessian matrix for testing
            gptq.H = torch.eye(256, dtype=torch.float32)  # Identity matrix for stability
            
            # Create mock weight data
            W = torch.randn(512, 256, dtype=torch.float32)
            gptq.module_copy = W
            
            # Mock quantizer
            gptq.quantizer.find_params = lambda x, weight=False: None
            
            # Run quantization
            Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=blocksize)
            
            print(f"  ✓ SUCCESS: Duration={duration:.3f}s, Avg Loss={avg_loss:.6f}")
            print(f"    Output shape: {Q.shape}, Scale shape: {scale.shape}, Zero shape: {zero.shape}")
            print(f"    Groups: {len(g_idx)} unique groups")
            
            # Verify output shapes
            assert Q.shape == W.shape, f"Output shape mismatch: {Q.shape} vs {W.shape}"
            assert scale.shape[1] == 1, f"Scale shape should be (n_groups, 1), got {scale.shape}"
            assert zero.shape[1] == 1, f"Zero shape should be (n_groups, 1), got {zero.shape}"
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Testing completed!")

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    
    print("\nTesting edge cases...")
    print("=" * 40)
    
    edge_cases = [
        # (group_size, blocksize, description)
        (-1, 128, "Channel-wise quantization (group_size=-1)"),
        (256, 128, "Large group_size > blocksize"),
        (8, 64, "Small group_size"),
        (128, 32, "blocksize < group_size"),
    ]
    
    for group_size, blocksize, description in edge_cases:
        print(f"\nTesting edge case: {description}")
        
        try:
            # Create test model
            model = create_test_model()
            
            # Create quantization config
            qcfg = QuantizeConfig(
                bits=4,
                group_size=group_size,
                block_size=blocksize,
                sym=True,
                desc_act=False,
                fast_loop=True
            )
            
            # Create GPTQ wrapper
            gptq = GPTQ(model, qcfg)
            
            # Create mock Hessian matrix
            gptq.H = torch.eye(256, dtype=torch.float32)
            
            # Create mock weight data
            W = torch.randn(512, 256, dtype=torch.float32)
            gptq.module_copy = W
            
            # Mock quantizer
            gptq.quantizer.find_params = lambda x, weight=False: None
            
            # Run quantization
            Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=blocksize)
            
            print(f"  ✓ SUCCESS: Handled gracefully")
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")

if __name__ == "__main__":
    test_quantization_configs()
    test_edge_cases()
    print("\nAll tests completed!")