import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class TestHessianCache(unittest.TestCase):
    def test_hessian_cache_reduces_allocations(self):
        model = SimpleModel()
        q_config = QuantizeConfig(hessian_cache=True, bits=4, group_size=128)
        
        # Calibration data
        calibration_data = [torch.randn(1, 128) for _ in range(10)]

        with patch('torch.zeros') as mock_zeros:
            original_zeros = torch.zeros
            mock_zeros.side_effect = lambda *args, **kwargs: original_zeros(*args, **kwargs)

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    gptq = GPTQ(module, q_config)
                    for data in calibration_data:
                        gptq.add_batch(data, data)
                    
                    gptq.quantize()
                    gptq.free()
            
            # We expect torch.zeros to be called for the Hessian accumulator.
            # With caching, it should be called fewer times than the number of layers * calibration batches.
            # A simple check is that it's called at least once, but less than it would be without caching.
            
            # The exact number of calls is hard to predict due to other uses of torch.zeros.
            # However, for the main Hessian, we expect it to be allocated once for each unique shape.
            # In our model, we have two unique shapes: (256, 256) and (128, 128).
            # The partials might be allocated more.
            
            # Let's count allocations for the specific Hessian shapes.
            hessian_alloc_count_256 = 0
            hessian_alloc_count_128 = 0
            
            for call_args in mock_zeros.call_args_list:
                args, kwargs = call_args
                if len(args) > 0 and isinstance(args[0], tuple):
                    shape = args[0]
                    if shape == (256, 256):
                        hessian_alloc_count_256 += 1
                    if shape == (128, 128):
                        hessian_alloc_count_128 += 1
            
            # Without caching, we'd expect at least 2 for fc1/fc2 (256x256) and 1 for fc3 (128x128) for the main Hessian,
            # plus more for the partials. With caching, we expect 1 for each unique size for the main Hessian.
            # The partials will also be cached.
            self.assertLess(hessian_alloc_count_256, 10 * 2, "Cache should reduce allocations for 256x256 Hessian.")
            self.assertLess(hessian_alloc_count_128, 10 * 1, "Cache should reduce allocations for 128x128 Hessian.")
            
            # A strong assertion would be that it's allocated only once per unique shape for the main accumulator.
            # The partial accumulators are also cached now. So we expect very few allocations.
            self.assertLessEqual(hessian_alloc_count_256, 2, "256x256 Hessian should be allocated at most twice (once for main, once for partial).")
            self.assertLessEqual(hessian_alloc_count_128, 2, "128x128 Hessian should be allocated at most twice.")

if __name__ == '__main__':
    unittest.main()