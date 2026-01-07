import unittest
from unittest.mock import MagicMock, patch, call
from gptqmodel.quantization.config import GcMode
import torch
from gptqmodel.looper.stage_subset import run_subset_stage, SubsetStageResult

class TestMoEExpertBatching(unittest.TestCase):
    def setUp(self):
        self.looper = MagicMock()
        self.processor = MagicMock()
        self.module = MagicMock()
        self.layer_inputs = [MagicMock()]
        self.layer_input_kwargs = [MagicMock()]
        self.position_ids = [MagicMock()]
        self.attention_masks = [MagicMock()]
        self.cur_layer_device = torch.device("cpu")
        self.full = {}
        self.shared_kv_cache_dict = {}
        self.pb = MagicMock()
        
        # Setup config
        self.looper.gptq_model.quantize_config.moe_bypass_router_experts_batch_size = None
        self.looper.gptq_model.quantize_config.gc_mode = GcMode.ON_STAGE_END
        self.looper.gptq_model.quantize_config.auto_forward_data_parallel = True
        
        # Setup mocks
        self.looper._is_attention_module_name.return_value = False
        self.looper._extract_moe_group_key.return_value = "moe.experts"
        self.looper._moe_subset_threshold = 2
        
        self.processor.name.return_value = "GPTQProcessor"
        self.processor.require_fwd = True
        
        # Create fake subset
        self.subset_names = [f"expert.{i}" for i in range(10)]
        self.subset = {name: MagicMock() for name in self.subset_names}
        self.looper.crate_named_modules.return_value = self.subset

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_no_batching(self, mock_empty_cache):
        # Default behavior: moe_bypass_router_experts_batch_size is None
        self.looper.gptq_model.quantize_config.moe_bypass_router_experts_batch_size = None
        
        # Need to patch the internal processing part if we want to separate it, 
        # but for now we test that it runs as one big block if we don't refactor yet,
        # or we assume refactoring.
        # Ideally, we mock the newly created '_run_single_subset_pass' if we had it.
        # Since we haven't written the code yet, this test will fail or error if run against current code 
        # because the function doesn't exist or logic differs.
        # I will structure this test to check behaviour AFTER refactoring.
        
        # Since I cannot mock inner functions easily without them existing, I will rely on checking
        # how many times 'looper._run_forward_batches' is called.
        
        self.looper._run_forward_batches.return_value = [torch.tensor([1.0])]
        self.looper._resolve_batch_total.return_value = 1
        self.looper._collect_row_counts.return_value = [1] 
        
        run_subset_stage(
            looper=self.looper,
            processor=self.processor,
            module=self.module,
            layer_inputs=self.layer_inputs,
            layer_input_kwargs=self.layer_input_kwargs,
            position_ids=self.position_ids,
            attention_masks=self.attention_masks,
            cur_layer_device=self.cur_layer_device,
            is_lm_head_module=False,
            layer_descriptor="layer.0",
            layer_title="title",
            layer_index=0,
            layers_prefix="model.layers",
            subset_names=self.subset_names,
            subset_index=0,
            subset_total=1,
            full=self.full,
            failsafe=False,
            shared_kv_cache_dict=self.shared_kv_cache_dict,
            pb=self.pb
        )
        
        # Should be called once for the whole subset
        self.assertEqual(self.looper._run_forward_batches.call_count, 1)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_expert_batching(self, mock_empty_cache):
        # Enable batching
        self.looper.gptq_model.quantize_config.moe_bypass_router_experts_batch_size = 2
        self.looper.gptq_model.quantize_config.gc_mode = GcMode.ON_STAGE_END
        
        self.looper._run_forward_batches.return_value = [torch.tensor([1.0])]
        self.looper._resolve_batch_total.return_value = 1
        self.looper._collect_row_counts.return_value = [1]
        
        # Setup experts with multiple modules per expert to test grouping
        # Expert 0: gate, up
        # Expert 1: gate, up
        # ...
        # Expert 9: gate, up
        # Total modules: 20
        # Expert Groups: 10
        # Batch Size: 2 (Experts) -> 5 batches
        
        subset_names = []
        subset = {}
        for i in range(10):
            gate_name = f"model.layers.0.experts.{i}.gate_proj"
            up_name = f"model.layers.0.experts.{i}.up_proj"
            subset_names.extend([gate_name, up_name])
            subset[gate_name] = MagicMock()
            subset[up_name] = MagicMock()
            
        self.looper.crate_named_modules.return_value = subset
        
        # Mock group key extraction
        def get_group_key(name):
            parts = name.split('.')
            if "experts" in parts:
                idx = parts.index("experts")
                return f"{'.'.join(parts[:idx+2])}" # e.g. model.layers.0.experts.0
            return None
        self.looper._extract_moe_group_key.side_effect = get_group_key
        
        run_subset_stage(
            looper=self.looper,
            processor=self.processor,
            module=self.module,
            layer_inputs=self.layer_inputs,
            layer_input_kwargs=self.layer_input_kwargs,
            position_ids=self.position_ids,
            attention_masks=self.attention_masks,
            cur_layer_device=self.cur_layer_device,
            is_lm_head_module=False,
            layer_descriptor="layer.0",
            layer_title="title",
            layer_index=0,
            layers_prefix="model.layers",
            subset_names=subset_names,
            subset_index=0,
            subset_total=1,
            full=self.full,
            failsafe=False,
            shared_kv_cache_dict=self.shared_kv_cache_dict,
            pb=self.pb
        )
        
        # With 10 expert groups and batch size 2, we expect 5 calls
        self.assertEqual(self.looper._run_forward_batches.call_count, 5)
        
        # And cleanup should be called 5 times
        self.assertEqual(mock_empty_cache.call_count, 5)

if __name__ == '__main__':
    unittest.main()
