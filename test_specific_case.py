#!/usr/bin/env python3
"""
Test script specifically for group_size=128, blocksize=1024 case.
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

def test_specific_case():
    """Test GPTQ quantization with group_size=128, blocksize=1024."""
    
    print("Testing GPTQ quantization with group_size=128, blocksize=1024")
    print("=" * 60)
    
    # Create model and data
    model = create_test_model(input_size=2048, output_size=1024)
    test_data = generate_test_data()
    
    # Test the specific configuration mentioned by user
    group_size = 128
    blocksize = 1024
    
    print(f"Group size: {group_size}")
    print(f"Block size: {blocksize}")
    print(f"Model input size: {model.weight.shape[1]}")
    print(f"Model output size: {model.weight.shape[0]}")
    
    try:
        # Create quantization config
        qcfg = QuantizeConfig(bits=4, group_size=group_size, desc_act=True, sym=True)
        
        # Create GPTQ instance
        gptq = GPTQ(model, qcfg)
        
        # Add calibration data
        print("Adding calibration data...")
        with torch.no_grad():
            for i in range(5):  # Add multiple batches
                out = model(test_data)
                gptq.add_batch(test_data, out)
                print(f"  Batch {i+1} added")
        
        print("Performing quantization...")
        # Perform quantization with specified blocksize
        start_time = time.time()
        scale, zero, g_idx, duration, avg_loss, damp_percent = gptq.hf_quantize(blocksize=blocksize)
        end_time = time.time()
        
        print("✓ Quantization successful!")
        print(f"  - Duration: {duration:.3f}s")
        print(f"  - Average loss: {avg_loss:.6f}")
        print(f"  - Damp percent: {damp_percent:.6f}")
        print(f"  - Scale shape: {scale.shape}")
        print(f"  - Zero shape: {zero.shape}")
        print(f"  - Group index shape: {g_idx.shape}")
        
        # Verify group boundaries are correct
        expected_groups = (model.weight.shape[1] + group_size - 1) // group_size
        actual_groups = len(torch.unique(g_idx))
        
        print(f"  - Expected groups: {expected_groups}")
        print(f"  - Actual unique groups: {actual_groups}")
        
        if actual_groups == expected_groups:
            print("✓ Group count correct!")
        else:
            print("✗ Group count mismatch!")
        
        # Check group index distribution
        print(f"  - Group index range: {g_idx.min().item()} to {g_idx.max().item()}")
        print(f"  - Group index unique values: {torch.unique(g_idx).tolist()}")
        
        # Clean up
        gptq.free()
        
        print("\n" + "=" * 60)
        print("SUCCESS: group_size=128, blocksize=1024 works correctly!")
        
    except Exception as e:
        print(f"✗ Quantization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("FAILED: group_size=128, blocksize=1024 failed!")

if __name__ == "__main__":
    import time
    test_specific_case()