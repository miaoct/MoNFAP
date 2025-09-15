# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from typing import Any, Optional, Tuple, Type, List
import torch
from torch import nn
from torch.nn import functional as F
from .maskedtransformer import TwoWayTransformer
from ..base import Classifier1D

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim=[320, 160, 80, 40],
        mlp_dim=[2048, 1024, 512, 256],
        num_classes=2,
        attn_mask_thr=0.5,
        cls_dropout=0.0,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.mlp_dim = mlp_dim
        self.attn_mask_thr=attn_mask_thr
        self.num_mask_tokens = num_classes
        
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim[0])
        
        self.transformers = nn.ModuleList(
            [
                TwoWayTransformer(depth=2, embedding_dim=self.transformer_dim[i], mlp_dim=self.mlp_dim[i], num_heads=8,attention_downsample_rate=1)
                for i in range(len(transformer_dim))
            ]
        )

        self.pe_layers = nn.ModuleList(
            [
                PositionEmbeddingRandom(self.transformer_dim[i] // 2) for i in range(len(transformer_dim))
            ]
        )
        
        self.token_layers = nn.ModuleList(
            [
                MLP(self.transformer_dim[i], self.transformer_dim[i], self.transformer_dim[i+1], 1)
                for i in range(len(transformer_dim)-1)
            ]
        )

        self.upscaling_layers = nn.ModuleList()
        for i in range(len(transformer_dim)-1):
            self.upscaling_layers.append(UpConv(self.transformer_dim[i], self.transformer_dim[i] // 2))

        self.classifer = Classifier1D(self.transformer_dim[-1], num_classes, cls_dropout)


    def forward(
        self,
        image_embeddings,
        dense_prompt_embeddings,
        output_mask,
    ):
        output_tokens = self.mask_tokens.weight
        tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    
        for i in range(len(self.transformer_dim)):
            
            # NOTE: prediction is of higher-resolution [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(output_mask, size=dense_prompt_embeddings[i].shape[-2:], mode='bilinear', align_corners=False)
            # must use bool type, if a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, 8, 1, 1).flatten(0, 1) < self.attn_mask_thr).bool()
            attn_mask = attn_mask.detach()
            
            src = src + dense_prompt_embeddings[i]
            
            image_pe = self.pe_layers[i](src.shape[-2:]).unsqueeze(0)
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
            b, c, h, w = src.shape
            
            # Run the transformer
            tokens, src = self.transformers[i](image_embedding=src, image_pe=pos_src, point_embedding=tokens, memory_mask=attn_mask)
            src = src.transpose(1, 2).view(b, c, h, w)
            
            # if the backbone is the HRNet or ViT, the below code is removed
            if i < 3:
                tokens = self.token_layers[i](tokens)
                # Upscale src
                src = self.upscaling_layers[i](src, dense_prompt_embeddings[i+1].shape[-2:])

        b, c, h, w = src.shape
        masks = (tokens @ src.view(b, c, h * w)).view(b, -1, h, w)

        # Generate image-level predictions
        cls_pred = self.classifer(tokens)

        return masks, cls_pred


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LayerNorm2d(out_channels*4),
            nn.GELU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        
    '''forward'''
    def forward(self, x, prompt_size):
        x = self.conv1(x)
        out = F.interpolate(x, size=prompt_size, mode='bilinear', align_corners=False)
        out = self.conv2(out)
        return out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
