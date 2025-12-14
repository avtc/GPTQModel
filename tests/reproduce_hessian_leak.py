
import torch
import torch.nn as nn
from gptqmodel.quantization.hessian_cache import HessianCache
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization import QuantizeConfig
import gc

class MockModule(nn.Linear):
    def __init__(self, size):
        super().__init__(size, size, bias=False)

def test_cache_growth():
    size = 128
    device = torch.device("cpu") # Use CPU for simple testing, logic is same
    
    # Mock Config
    qcfg = QuantizeConfig()
    qcfg.hessian_cache = True
    
    module = MockModule(size)
    gptq = GPTQ(module, qcfg)
    
    # Simulate batches
    batch_size = 16
    n_batches = 50
    
    print(f"Initial Cache Size: {len(HessianCache().cache)}")
    
    for i in range(n_batches):
        inp = torch.randn(1, batch_size, size, device=device)
        # Dummy output
        out = torch.randn(1, batch_size, size, device=device)
        
        gptq.add_batch(inp, out)
        
        # Check cache
        cache = HessianCache().cache
        total_items = sum(len(v) for v in cache.values())
        print(f"Batch {i}: Cache Items: {total_items}")
        
    final_items = sum(len(v) for v in HessianCache().cache.values())
    print(f"Final Cache Items: {final_items}")
    
    if final_items > 5: # Expect 1 or 2 (one acc, one xtx cycling)
        print("FAIL: Cache grew too large")
    else:
        print("PASS: Cache stable")

if __name__ == "__main__":
    test_cache_growth()
