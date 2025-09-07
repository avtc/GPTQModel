# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, move_to
from .._const import CPU, EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


class GLM4VMoEGPTQ(BaseGPTQModel):
    loader = AutoModelForImageTextToText

    # GLM-4V MoE Model Structure:
    # Layer 0: Standard MLP (no MoE experts) - handled by ["mlp.down_proj"], ["mlp.gate_proj"], ["mlp.up_proj"]
    # Layers 1-46: MoE with shared_experts and individual experts (128 experts total) - handled by MoE components
    # Layer 46: Additional special structure with expanded parameters (embed_tokens, shared_head, eh_proj, etc.)
    #   This is handled dynamically through layer_modules_strict = False
    #
    # allow dynamic expert index for layer_modules so we don't need to write out 128 layers here
    # config.n_routed_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    base_modules = [
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.visual",
        "lm_head.weight"
    ]
    pre_lm_head_norm_module = "model.language_model.norm"

    # Set to False since GLM-4V MoE may have dynamic module structures
    layer_modules_strict = False

    layers_node = ["model.language_model.layers"]
    layer_type = "GLM4MoEDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],

        # MoE components for layers 1-46 (all have shared_experts and individual experts)
        ["mlp.shared_experts.up_proj", "mlp.shared_experts.gate_proj"],
        ["mlp.shared_experts.down_proj"],
        ["mlp.gate"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj", f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],

        # Standard MLP components for layer 0 (no experts)
        ["mlp.down_proj"],
        ["mlp.gate_proj"],
        ["mlp.up_proj"],
    ]

    layers_modules_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": {
                "shared_experts": ("up_proj", "gate_proj", "down_proj"),
                "gate": ("gate",),
                "experts": {
                    "#": ("up_proj", "gate_proj", "down_proj"),
                },
                # Standard MLP components for layer 0 (no experts)
                "down_proj": ("down_proj",),
                "gate_proj": ("gate_proj",),
                "up_proj": ("up_proj",),
            },
        }
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        pass

    @staticmethod
    def process_vision_info(
            conversations: list[dict] | list[list[dict]],
    ) -> Optional[list[Image.Image]]:
        vision_infos = extract_vision_info(conversations)
        # Read images
        image_inputs = []
        has_vision_content = False
        
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                has_vision_content = True
                image_inputs.append(fetch_image(vision_info))
        
        # Only raise error if vision content is expected but not found
        # For text-only quantization, this should be allowed
        if not has_vision_content:
            image_inputs = None
        
        return image_inputs

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(self, calibration_dataset, calibration_dataset_concat_size=None, batch_size: int = 1):
        processor = self.load_processor()
        calib_data = []
        
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            # Handle different input formats intelligently
            if isinstance(batch[0], str):
                # Convert string list to conversation format
                batch_conversations = [
                    [{"role": "user", "content": [{"type": "text", "text": text}]}]
                    for text in batch
                ]
            else:
                batch_conversations = batch
            
            # Apply chat template
            text = processor.apply_chat_template(
                batch_conversations, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info (handles text-only gracefully now)
            image_inputs = self.process_vision_info(batch_conversations)
            
            # Create inputs
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        
        del processor
        return calib_data