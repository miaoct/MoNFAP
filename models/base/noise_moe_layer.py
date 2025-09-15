import torch
import torch.nn as nn

from .importance_loss import importance_loss, importance
from .srm_conv import SRMConv2d_Separate
from .bayar_conv import BayarConv2d
from .cd_conv import Conv2d_cd
from .hf_conv import HFConv2d

class NoiseMoELayer(nn.Module):

    def __init__(self, gate, channels, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [
                Conv2d_cd(in_channels=channels, out_channels=channels),
                BayarConv2d(in_channels=channels, out_channels=channels, padding=2),
                SRMConv2d_Separate(inc=channels, outc=channels, learnable=True),
                HFConv2d(inc=channels, outc=channels, learnable=True)
            ]
        )
        self.shared_expert = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gate = gate
        
    def _get_device(self):
        # return self.shared_expert.weight.device
        return self.gate._get_device()

    def forward(self, x):
        identity = x
        
        weights = self.gate.compute_gating(x)
        examples_per_expert = (weights > 0).sum(dim=0)
        expert_importance = importance(weights)
        aux_loss = self.gate.compute_loss(weights)
        mask = weights > 0
        results = []
        for i in range(self.num_experts):
            # select mask according to computed gates (conditional computing)
            mask_expert = mask[:, i]
            # apply mask to inputs
            expert_input = x[mask_expert]
            # compute outputs for selected examples
            expert_output = self.experts[i](expert_input).to(self._get_device())
            # calculate output shape
            output_shape = list(expert_output.shape)
            output_shape[0] = x.size()[0]
            # store expert results in matrix
            expert_result = torch.zeros(output_shape, device=self._get_device())
            expert_result[mask_expert] = expert_output
            # weight expert's results
            expert_weight = weights[:, i].reshape(
                expert_result.shape[0], 1, 1, 1).to(self._get_device())
            expert_result = expert_weight * expert_result
            results.append(expert_result)
        # Combining results
        out = self.shared_expert(identity) + torch.stack(results, dim=0).sum(dim=0)
        
        return {'output': out,
                'aux_loss': aux_loss,
                'examples_per_expert': examples_per_expert,
                'expert_importance': expert_importance,
                'weights': weights}
