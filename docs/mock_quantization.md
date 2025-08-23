# Mock Quantization for Fast Model Loading Tests

## Overview

The mock quantization feature allows you to skip the expensive GPTQ quantization computations and quickly create a GPTQ-compatible model format. This is extremely useful for:

- Quickly checking if a model can be loaded by vllm/sglang
- Testing model compatibility without waiting for full quantization
- Development and debugging workflows
- CI/CD pipelines where you need to verify model loading

## Usage

### Enable Mock Quantization

Add `mock_quantization=True` to your `QuantizeConfig`:

```python
from gptqmodel import GPTQModel
from gptqmodel.quantization import QuantizeConfig

# Enable mock quantization for fast testing
quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    mock_quantization=True,  # This is the key option to enable mock mode
    desc_act=True,
    sym=True
)

model = GPTQModel.from_pretrained(
    model_id_or_path="your-model-id",
    quantize_config=quantize_config
)
```

### Behavior

When `mock_quantization=True`:

1. **Input Processing**: Skips expensive input data processing and calibration
2. **Hessian Computation**: Skips the expensive Hessian matrix computation
3. **Quantization Loops**: Skips the main quantization loops
4. **Result Generation**: Creates simulated GPTQ-compatible weights and metadata:
   - **Simulated Quantized Weights**: Creates weights that appear to be quantized at the specified `bits` and `group_size` without the expensive computations
   - **Realistic Scale/Zero Tensors**: Generates fake scale and zero tensors with proper quantization characteristics
   - **Proper Group Indices**: Creates appropriate group indices based on the specified `group_size`
   - **Realistic Quantization Metrics**: Generates fake but realistic-looking quantization metrics
   - **Minimal Execution Time**: Completes in milliseconds instead of minutes/hours

### Performance

- **Normal Quantization**: Minutes to hours depending on model size
- **Mock Quantization**: Seconds to milliseconds

### Example

```python
import time
from gptqmodel import GPTQModel
from gptqmodel.quantization import QuantizeConfig

# Fast mock quantization
config = QuantizeConfig(bits=4, mock_quantization=True)
model = GPTQModel.from_pretrained("model-id", quantize_config=config)

# Very fast quantization (milliseconds instead of minutes)
start = time.time()
model.quantize(["Sample input data..."])
print(f"Quantization completed in {time.time() - start:.4f} seconds")

# The model is now in GPTQ format and can be loaded by vllm/sglang
model.save_quantized("output-path")
```

## Use Cases

### 1. Quick Compatibility Testing

```python
# Check if your model works with vllm/sglang quickly
config = QuantizeConfig(mock_quantization=True)
model = GPTQModel.from_pretrained("your-model", quantize_config=config)
model.save_quantized("gptq-model")

# Now test with vllm/sglang loading
```

### 2. Development Workflow

During development, you can use mock quantization to quickly test model loading and integration before running the full quantization process.

### 3. CI/CD Pipeline

In automated pipelines, use mock quantization to quickly verify that the model loading process works before committing to expensive full quantization.

## Important Notes

- **Model Accuracy**: Mock quantization does not actually quantize the model. The weights remain unchanged in their original precision.
- **Compatibility**: The output is in GPTQ-compatible format and can be loaded by vllm/sglang and other inference engines.
- **Intended Use**: This is designed for testing and development, not for production deployment.
- **Metrics**: All quantization metrics (loss, time, etc.) are fake and should not be used for actual evaluation.

## Configuration Options

You can combine mock quantization with other GPTQModel options:

```python
config = QuantizeConfig(
    bits=4,                    # Bit width for simulated quantization
    group_size=128,           # Group size for simulated quantization 
    mock_quantization=True,   # Enable mock mode
    desc_act=True,           # Activation desc still processed for compatibility
    sym=True,                # Symmetric quantization setting
    # ... other options
)
```

## Comparison

| Feature | Normal Quantization | Mock Quantization |
|---------|-------------------|------------------|
| Execution Time | Minutes to hours | Seconds to milliseconds |
| Model Accuracy | Actual quantization | Simulated quantization at specified bits/group_size |
| GPTQ Compatibility | Full compatibility | Full compatibility |
| Use Case | Production deployment | Testing and development |
| File Size | Significantly smaller | Same as original model |

## Technical Details

### Mock Quantization Process

When `mock_quantization=True` is enabled, the system:

1. **Skips Expensive Operations**: Bypasses Hessian computation, input processing, and main quantization loops
2. **Simulates Quantization**: Creates weights that appear quantized at the specified `bits` and `group_size`:
   - For per-group quantization: Creates group-wise quantized appearance
   - For per-tensor quantization: Creates tensor-wise quantized appearance
   - Generates realistic scale/zero parameters matching the quantization settings
3. **Maintains Compatibility**: Ensures the output format is fully compatible with GPTQ inference engines

### Realism vs Performance

The mock quantization provides a balance between:
- **Realism**: Simulates actual quantization characteristics at the specified settings
- **Performance**: Executes in milliseconds instead of hours
- **Compatibility**: Maintains full GPTQ format compatibility

## Troubleshooting

### Mock Quantization Not Working

If mock quantization doesn't seem to be working:

1. Verify `mock_quantization=True` is set in `QuantizeConfig`
2. Check that you're using the latest version of GPTQModel
3. Ensure no other quantization options are overriding the mock behavior

### Loading Issues

If the mock-quantized model doesn't load properly:

1. Verify the model architecture is supported by GPTQModel
2. Check that the output directory contains proper GPTQ files
3. Test with a smaller model first to verify the workflow

## Conclusion

Mock quantization provides a fast way to create GPTQ-compatible models for testing and development purposes. It's particularly valuable for verifying model compatibility with inference engines like vllm and sglang without the time cost of full quantization. The enhanced implementation now properly simulates quantized weights at the specified bits and group_size settings, providing a more realistic testing experience.