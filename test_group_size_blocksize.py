#!/usr/bin/env python3
"""
Test script to verify that GPTQ quantization works properly with different group_size and blocksize values.
"""

import torch
import torch.nn as nn
import numpy as np
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization import QuantizeConfig

def create_test_model(input_size=2048, output_size=1024):
    """Create a simple linear model for testing."""
    return nn.Linear(input_size, output_size, bias=False)

def generate_test_data(batch_size=32, seq_len=128, input_size=2048):
    """Generate test data for calibration."""
    return torch.randn(batch_size, seq_len, input_size)

def test_gptq_quantization():
    """Test GPTQ quantization with different group_size and blocksize combinations."""
    
    # Test configurations
    test_configs = [
        {"group_size": 128, "blocksize": 128},   # Same size
        {"group_size": 128, "blocksize": 256},   # Different sizes
        {"group_size": 128, "blocksize": 512},   # Different sizes  
        {"group_size": 128, "blocksize": 1024},  # Different sizes (user's example)
        {"group_size": 64, "blocksize": 256},    # Different sizes
        {"group_size": 256, "blocksize": 128},   # Different sizes
    ]
    
    model = create_test_model()
    test_data = generate_test_data()
    
    print("Testing GPTQ quantization with different group_size and blocksize combinations...")
    print("=" * 80)
    
    for config in test_configs:
        group_size = config["group_size"]
        blocksize = config["blocksize"]
        
        print(f"\nTesting: group_size={group_size}, blocksize={blocksize}")
        print("-" * 50)
        
        try:
            # Create quantization config
            qcfg = QuantizeConfig(bits=4, group_size=group_size, desc_act=True, sym=True)
            
            # Create GPTQ instance
            gptq = GPTQ(model, qcfg)
            
            # Add calibration data
            with torch.no_grad():
                for _ in range(5):  # Add multiple batches
                    out = model(test_data)
                    gptq.add_batch(test_data, out)
            
            # Perform quantization with specified blocksize
            start_time = time.time()
            scale, zero, g_idx, duration, avg_loss, damp_percent = gptq.hf_quantize(blocksize=blocksize)
            end_time = time.time()
            
            print(f"✓ Quantization successful!")
            print(f"  - Duration: {duration:.3f}s")
            print(f"  - Average loss: {avg_loss:.6f}")
            print(f"  - Damp percent: {damp_percent:.6f}")
            print(f"  - Scale shape: {scale.shape}")
            print(f"  - Zero shape: {zero.shape}")
            print(f"  - Group index shape: {g_idx.shape}")
            print(f"  - Unique groups: {len(torch.unique(g_idx))}")
            
            # Verify group boundaries are correct
            expected_groups = (model.weight.shape[1] + group_size - 1) // group_size
            actual_groups = len(torch.unique(g_idx))
            
            if actual_groups == expected_groups:
                print(f"✓ Group count correct: expected {expected_groups}, got {actual_groups}")
            else:
                print(f"✗ Group count mismatch: expected {expected_groups}, got {actual_groups}")
            
            # Clean up
            gptq.free()
            
        except Exception as e:
            print(f"✗ Quantization failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")

if __name__ == "__main__":
    import time
    test_gptq_quantization()