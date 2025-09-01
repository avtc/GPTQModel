# Memory Optimization Feature for GPTQModel

## Overview

The memory optimization feature in GPTQModel addresses the issue of excessive RAM usage during model quantization, especially for large models. When enabled, this feature significantly reduces memory consumption by immediately freeing source layers after quantization, allowing only one source layer and one quantized layer to be held in memory at a time.

## Problem Statement

During quantization, the transformers library has a known memory leak (huggingface/transformers#34366) where mmaped memory is not released properly. This causes:

1. **High memory usage**: Source layers remain loaded throughout the entire quantization process
2. **Limited scalability**: Large models cannot be quantized on systems with limited RAM
3. **Memory fragmentation**: mmaped memory chunks are held even when not in use

## Solution

The memory optimization feature implements several strategies:

1. **Immediate source layer cleanup**: After each layer is quantized, the source layer is immediately freed from memory
2. **Enhanced garbage collection**: Strategic torch_empty_cache() calls to clean up GPU/PyTorch cache
3. **Memory-efficient processing**: Only one layer (source + quantized) is held in memory at any time

## Usage

### Basic Usage

```python
from gptqmodel import GPTQModel
from gptqmodel.quantization.config import QuantizeConfig

# Enable memory optimization
config = QuantizeConfig(
    bits=4,
    group_size=128,
    memory_optimization=True  # Enable memory optimization
)

# Load model and quantize with memory optimization
model = GPTQModel.load("your-model-name")
model.quantize(
    tokenizer=tokenizer,
    calib_data=calibration_dataset,
    quantize_config=config
)
```

### Advanced Usage

```python
# Full example with memory optimization
import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel
from gptqmodel.quantization.config import QuantizeConfig

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("big-model-name")
model = GPTQModel.load("big-model-name")

# Create calibration dataset
calibration_dataset = [
    {"input_ids": tokenizer("Sample text", return_tensors="pt").input_ids[0]}
    for _ in range(100)
]

# Configure with memory optimization
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    memory_optimization=True,  # Enable memory optimization
    desc_act=True,
    sym=True
)

# Quantize with memory optimization - suitable for large models
model.quantize(
    tokenizer=tokenizer,
    calib_data=calibration_dataset,
    quantize_config=quant_config
)

# Save the quantized model
model.save_pretrained("output-directory")
```

## Configuration Options

### QuantizeConfig Parameters

- `memory_optimization: bool` (default: `False`)
  - When `True`, enables memory optimization during quantization
  - Reduces peak memory usage by immediately freeing source layers
  - Recommended for large models or systems with limited RAM

### Memory Optimization Behavior

When `memory_optimization=True`:

1. **Layer Processing**: Each layer is processed one-by-one
2. **Immediate Cleanup**: Source layer memory is freed immediately after quantization
3. **Garbage Collection**: Strategic torch_empty_cache() calls are made
4. **Memory Efficiency**: Only current layer (source + quantized) is held in memory

### Performance Considerations

- **Memory Usage**: Significantly reduced peak memory consumption
- **Speed**: Minimal performance overhead (slight increase due to cleanup operations)
- **Compatibility**: Works with all existing quantization methods (GPTQ, AutoRound, QQQ)
- **Accuracy**: No impact on quantization accuracy

## Benefits

### 1. Reduced Memory Footprint
- **Before**: All source layers remain loaded throughout quantization
- **After**: Only one source layer + one quantized layer in memory at a time

### 2. Improved Scalability
- Enables quantification of larger models on systems with limited RAM
- Makes quantization accessible to users with consumer-grade hardware

### 3. Better Memory Management
- Addresses transformers library memory leak issues
- More efficient memory utilization during quantization

### 4. Configurable
- Can be enabled/disabled based on system requirements
- Backward compatible with existing workflows

## Technical Implementation

### Key Components

1. **QuantizeConfig**: Added `memory_optimization` field
2. **GPTQProcessor**: Enhanced cleanup after layer processing
3. **ModuleLooper**: Strategic cache clearing during quantization loop
4. **torch_utils**: Enhanced memory cleanup utilities
5. **ModelLoader**: Layer-wise model loading functionality

### Enhanced Memory Optimization Flow

#### Standard Mode
```python
# Simplified flow showing basic memory optimization
for layer in layers:
    # Load and process current layer
    g = GPTQ(layer, config)
    wq, scale, zero, g_idx = g.quantize()
    
    # IMMEDIATE cleanup when memory_optimization=True
    if config.memory_optimization:
        g.free()  # Free source layer immediately
        del g.H, g.module_copy
        torch_empty_cache()  # Force cleanup
    
    # Save quantized results
    save_quantized_results(layer, wq, scale, zero, g_idx)
```

#### Enhanced Mode (Layer-wise Loading)
```python
# Enhanced flow with layer-wise model loading
if config.memory_optimization:
    # Create model skeleton
    model = create_model_skeleton(config)
    
    for layer_index in range(num_layers):
        # Load only current layer
        layer = load_layer_skeleton(model, layer_index)
        
        # Process layer
        g = GPTQ(layer, config)
        wq, scale, zero, g_idx = g.quantize()
        
        # IMMEDIATE cleanup
        g.free()
        del layer
        torch_empty_cache()
        
        # Save quantized results
        save_quantized_results(layer_index, wq, scale, zero, g_idx)
```

### True Layer-by-Layer Loading Implementation

The enhanced layer-wise loading approach:

1. **Model Skeleton Creation**: Creates a model architecture without weights
2. **Individual Layer Loading**: Uses `_load_single_layer()` to load only the specific layer needed for quantization
3. **Minimal Memory Footprint**: Only loads the current layer + its modules from the checkpoint
4. **Immediate Cleanup**: Layer memory is automatically freed after processing
5. **Iterative Processing**: Each layer is loaded, processed, and freed before moving to the next

This approach dramatically reduces memory usage by:
- Never loading the entire model at once
- Loading only the current layer needed for quantization
- Automatically cleaning up after each layer is processed
- Maintaining only one source layer + one quantized layer in memory at any time

### Technical Implementation

```python
# True layer-by-layer loading flow
for layer_index in range(num_layers):
    # Load ONLY the specific layer needed for quantization
    layer = ModelLoader._load_single_layer(
        model_local_path=model_path,
        config=model_config,
        layers_prefix=layers_node,
        layer_index=layer_index,
        layer_modules=layer_modules
    )
    
    # Process the layer
    g = GPTQ(layer, config)
    wq, scale, zero, g_idx = g.quantize()
    
    # Layer is automatically cleaned up by _load_single_layer
    # Only quantized results are kept
    save_quantized_results(layer_index, wq, scale, zero, g_idx)
```

## Testing

Run the included test script to verify memory optimization functionality:

```bash
python test_memory_optimization.py
```

The test script verifies:
- Configuration field is properly added
- Memory optimization flag is propagated correctly
- Quantization works with memory optimization enabled

## Troubleshooting

### Out of Memory Errors Still Occurring

If you still encounter OOM errors with `memory_optimization=True`:

1. **Reduce batch size**: Use smaller calibration datasets
2. **Use sequential processing**: Ensure `true_sequential=True` in config
3. **Check GPU memory**: Monitor GPU memory usage with `nvidia-smi`
4. **Consider smaller models**: For extremely large models, consider model partitioning

### Performance Impact

If performance is impacted:
1. **Monitor CPU usage**: Memory cleanup may increase CPU usage slightly
2. **Balance with memory**: Adjust based on your system's capabilities
3. **Test both modes**: Compare `memory_optimization=True/False` for your use case

## Future Enhancements

Planned improvements:

1. **Layer-wise Loading**: Load specific modules without loading entire model
2. **Adaptive Memory Management**: Dynamic adjustment based on available memory
3. **Memory Profiling**: Tools to monitor memory usage during quantization
4. **Multi-GPU Optimization**: Better memory distribution across multiple GPUs

## Conclusion

The memory optimization feature makes GPTQModel more accessible to users with limited hardware resources while maintaining quantization quality. By addressing the transformers library memory leak issues and implementing efficient layer-by-layer processing, users can now quantify larger models that were previously impossible due to memory constraints.

The feature is backward compatible and can be easily enabled by setting `memory_optimization=True` in the QuantizeConfig.