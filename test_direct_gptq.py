#!/usr/bin/env python3
"""
Direct test of GPTQ quantization with different group_size and blocksize values.
This test bypasses the problematic hf_quantize method and calls quantize() directly.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization import QuantizeConfig

def create_test_model(input_size=2048, output_size=1024):
    """Create a simple linear model for testing."""
    return nn.Linear(input_size, output_size, bias=False)

def generate_test_data(batch_size=32, seq_len=128, input_size=2048):
    """Generate test data for calibration."""
    return torch.randn(batch_size, seq_len, input_size)

def test_direct_gptq():
    """Test GPTQ quantization directly with different group_size and blocksize combinations."""
    
    print("Direct GPTQ Quantization Test")
    print("=" * 50)
    
    # Test configurations - compare fast_loop=True vs fast_loop=False
    test_configs = [
        # Baseline test (same size)
        {"group_size": 128, "blocksize": 128, "fast_loop": True, "name": "Fast: group_size=128, blocksize=128 (baseline)"},
        {"group_size": 128, "blocksize": 128, "fast_loop": False, "name": "Original: group_size=128, blocksize=128 (baseline)"},
        # Fast loop tests
        {"group_size": 128, "blocksize": 1024, "fast_loop": True, "name": "Fast: group_size=128, blocksize=1024"},
        {"group_size": 64, "blocksize": 256, "fast_loop": True, "name": "Fast: group_size=64, blocksize=256"},
        # Original loop tests for comparison
        {"group_size": 128, "blocksize": 1024, "fast_loop": False, "name": "Original: group_size=128, blocksize=1024"},
        {"group_size": 64, "blocksize": 256, "fast_loop": False, "name": "Original: group_size=64, blocksize=256"},
    ]
    
    # Create model and data
    model = create_test_model(input_size=2048, output_size=1024)
    test_data = generate_test_data()
    
    # Store original weights for comparison
    original_weights = model.weight.data.clone()
    
    results = []
    
    for config in test_configs:
        # Create fresh model for each test
        model.weight.data = original_weights.clone()
        
        print(f"\nTesting: {config['name']}")
        print(f"  - Group size: {config['group_size']}")
        print(f"  - Block size: {config['blocksize']}")
        print(f"  - Model input size: {model.weight.shape[1]}")
        print(f"  - Model output size: {model.weight.shape[0]}")
        
        try:
            # Create quantization config with proper structure
            qcfg = QuantizeConfig(
                bits=8,
                group_size=config['group_size'],
                v2=False,
                v2_memory_device="auto",
                parallel_packing=True,
                sym=True,
                desc_act=False,
                hyb_act=True,
                mock_hessian_inverse=False,
                fast_loop=config['fast_loop'],
                block_size=config['blocksize'],
            )
            
            # Create GPTQ instance
            gptq = GPTQ(model, qcfg)
            
            # Configure the quantizer properly (equivalent to what gptq_processor.py does)
            gptq.quantizer.configure(perchannel=True)
            
            # Add calibration data
            print("  Adding calibration data...")
            with torch.no_grad():
                for i in range(3):  # Add multiple batches
                    out = model(test_data)
                    gptq.add_batch(test_data, out)
                    print(f"    Batch {i+1} added")
            
            # Perform quantization directly using quantize() method
            print("  Performing quantization...")
            start_time = time.time()
            Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples = gptq.quantize(blocksize=config['blocksize'])
            end_time = time.time()
            
            print("  ✓ Quantization successful!")
            print(f"    - Duration: {duration:.3f}s")
            print(f"    - Average loss: {avg_loss:.6f}")
            print(f"    - Damp percent: {damp_percent:.6f}")
            print(f"    - Scale shape: {scale.shape}")
            print(f"    - Zero shape: {zero.shape}")
            print(f"    - Group index shape: {g_idx.shape}")
            
            # Verify group boundaries
            expected_groups = (model.weight.shape[1] + config['group_size'] - 1) // config['group_size']
            actual_groups = len(torch.unique(g_idx))
            
            print(f"    - Expected groups: {expected_groups}")
            print(f"    - Actual unique groups: {actual_groups}")
            
            if actual_groups == expected_groups:
                print("    ✓ Group count correct!")
            else:
                print("    ✗ Group count mismatch!")
                return False
            
            # Check group index range
            g_idx_min = g_idx.min().item()
            g_idx_max = g_idx.max().item()
            print(f"    - Group index range: {g_idx_min} to {g_idx_max}")
            
            # Verify group indices are sequential
            unique_groups = torch.unique(g_idx).sort().values
            expected_unique = torch.arange(expected_groups, device=g_idx.device)
            if torch.equal(unique_groups, expected_unique):
                print("    ✓ Group indices are sequential!")
            else:
                print("    ✗ Group indices are not sequential!")
                print(f"      Expected: {expected_unique.tolist()}")
                print(f"      Actual: {unique_groups.tolist()}")
                return False
            
            # Clean up
            gptq.free()
            results.append(True)
            
        except Exception as e:
            print(f"  ✗ Quantization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for i, result in enumerate(results):
        config = test_configs[i]
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {config['name']} (group_size={config['group_size']}, blocksize={config['blocksize']})")
    
    if passed == total:
        print("\n🎉 All tests passed! GPTQ quantization works correctly with different group_size and blocksize values.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    test_direct_gptq()