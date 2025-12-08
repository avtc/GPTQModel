# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class InputCache:
    layer_inputs: List[List[torch.Tensor]]
    layer_input_kwargs: List[Dict[str, torch.Tensor]]
    position_ids: List[torch.Tensor]
    attention_masks: List[torch.Tensor]

    def module_kwargs(self):
        result = dict()
        result["position_ids"] = self.position_ids
        result["attention_masks"] = self.attention_masks
        return result

    def detach(self):
        """
        Detaches all tensors in the cache from their computation graph.
        This is crucial for breaking references that can prevent garbage collection
        or block operations like `accelerate.disk_offload`.
        """
        for i in range(len(self.layer_inputs)):
            for j in range(len(self.layer_inputs[i])):
                if self.layer_inputs[i][j] is not None:
                    self.layer_inputs[i][j] = self.layer_inputs[i][j].detach()

        for i in range(len(self.layer_input_kwargs)):
            for key in self.layer_input_kwargs[i]:
                if isinstance(self.layer_input_kwargs[i][key], torch.Tensor):
                    self.layer_input_kwargs[i][key] = self.layer_input_kwargs[i][key].detach()

        for i in range(len(self.position_ids)):
            if self.position_ids[i] is not None:
                self.position_ids[i] = self.position_ids[i].detach()

        for i in range(len(self.attention_masks)):
            if self.attention_masks[i] is not None:
                self.attention_masks[i] = self.attention_masks[i].detach()
