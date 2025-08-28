#!/usr/bin/env python3
"""
Comprehensive test script to verify GPTQ quantization works properly with different group_size and blocksize combinations.
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

def run_test_case(model, test_data, group_size, blocksize, test_name):
    """Run a single test case."""
    print(f"\nTesting: {test_name}")
    print(f"  - Group size: {group_size}")
    print(f"  - Block size: {blocksize}")
    print(f"  - Model input size: {model.weight.shape[1]}")
    print(f"  - Model output size: {model.weight.shape[0]}")
    
    try:
        # Create quantization config
        qcfg = QuantizeConfig(bits=4, group_size=group_size, desc_act=True, sym=True)
        
        # Create GPTQ instance
        gptq = GPTQ(model, qcfg)
        
        # Add calibration data
        print("  Adding calibration data...")
        with torch.no_grad():
            for i in range(3):  # Add multiple batches
                out = model(test_data)
                gptq.add_batch(test_data, out)
                print(f"    Batch {i+1} added")
        
        # Perform quantization
        print("  Performing quantization...")
        start_time = time.time()
        scale, zero, g_idx, duration, avg_loss, damp_percent = gptq.hf_quantize(blocksize=blocksize)
        end_time = time.time()
        
        print("  ✓ Quantization successful!")
        print(f"    - Duration: {duration:.3f}s")
        print(f"    - Average loss: {avg_loss:.6f}")
        print(f"    - Damp percent: {damp_percent:.6f}")
        print(f"    - Scale shape: {scale.shape}")
        print(f"    - Zero shape: {zero.shape}")
        print(f"    - Group index shape: {g_idx.shape}")
        
        # Verify group boundaries
        expected_groups = (model.weight.shape[1] + group_size - 1) // group_size
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
        return True
        
    except Exception as e:
        print(f"  ✗ Quantization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive():
    """Test GPTQ quantization with various group_size and blocksize combinations."""
    
    print("Comprehensive GPTQ Quantization Test")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {"group_size": 128, "blocksize": 128, "name": "Same size (baseline)"},
        {"group_size": 128, "blocksize": 256, "name": "Different sizes"},
        {"group_size": 128, "blocksize": 512, "name": "Different sizes"},
        {"group_size": 128, "blocksize": 1024, "name": "User's example"},
        {"group_size": 64, "blocksize": 256, "name": "Different sizes"},
        {"group_size": 256, "blocksize": 128, "name": "Different sizes (reverse)"},
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
        
        success = run_test_case(
            model=model,
            test_data=test_data,
            group_size=config["group_size"],
            blocksize=config["blocksize"],
            test_name=config["name"]
        )
        
        results.append({
            "name": config["name"],
            "group_size": config["group_size"],
            "blocksize": config["blocksize"],
            "success": success
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for result in results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  {status}: {result['name']} (group_size={result['group_size']}, blocksize={result['blocksize']})")
    
    if passed == total:
        print("\n🎉 All tests passed! GPTQ quantization works correctly with different group_size and blocksize values.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    test_comprehensive()