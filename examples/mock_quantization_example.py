#!/usr/bin/env python3
"""
Example demonstrating the mock quantization feature for fast model loading tests.

This example shows how to use the mock_quantization option to quickly check
if a model can be loaded by vllm/sglang without waiting for full quantization.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gptqmodel import GPTQModel
from gptqmodel.quantization import QuantizeConfig

def main():
    print("=== Mock Quantization Example ===\n")
    
    # Example model - you can replace this with any supported model
    model_id = "facebook/opt-125m"  # Small model for quick testing
    
    print(f"Testing mock quantization with model: {model_id}")
    
    # Create quantize config with mock quantization enabled
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        mock_quantization=True,  # Enable mock quantization for fast testing
        desc_act=True,
        sym=True
    )
    
    print(f"Quantization config: mock_quantization={quantize_config.mock_quantization}")
    print("This will skip expensive quantization computations and create fake GPTQ-compatible weights.\n")
    
    try:
        # Load the model with mock quantization enabled
        print("Loading model with mock quantization...")
        start_time = time.time()
        
        model = GPTQModel.from_pretrained(
            model_id_or_path=model_id,
            quantize_config=quantize_config,
            device="auto"  # Use auto device detection
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds (very fast due to mock mode)")
        
        # Test that we can call quantize (it should be very fast)
        print("\nTesting quantize() method with mock data...")
        quant_start = time.time()
        
        # Create some dummy calibration data
        calibration_data = [
            "Hello world, this is a test sentence for quantization.",
            "GPTQModel supports mock quantization for fast testing.",
            "This feature helps quickly verify model compatibility with vllm/sglang."
        ]
        
        # This should be extremely fast due to mock mode
        quant_log = model.quantize(calibration_data)
        quant_time = time.time() - quant_start
        
        print(f"Quantization completed in {quant_time:.4f} seconds")
        print("Mock quantization successfully created GPTQ-compatible format!")
        
        # Save the model (optional - for testing)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "mock_quantized_model")
            model.save_quantized(save_path)
            print(f"\nMock quantized model saved to: {save_path}")
            print("You can now test loading this model with vllm/sglang")
        
        print("\n=== Summary ===")
        print("✓ Mock quantization enabled successfully")
        print("✓ Model loaded very quickly")
        print("✓ Quantization completed in milliseconds")
        print("✓ GPTQ-compatible format created")
        print("\nThe model is now ready for testing with vllm/sglang!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This example requires a model to be available.")
        print("You can replace 'facebook/opt-125m' with any supported model ID or local path.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())