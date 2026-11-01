"""ConvNeXt-style network built from MobileNetV4 Universal Inverted Bottleneck (UIB) blocks.

The UIB block itself comes from timm (`timm.models._efficientnet_blocks.UniversalInvertedResidual`).
Its two optional depth-wise convolutions select one of four variants:

    variant      dw_start  dw_mid   structure
    ---------    --------  ------   ------------------------------------------
    'convnext'      k        0      DW(k) -> PW expand -> PW project   (ConvNeXt block)
    'extra_dw'      k        3      DW(k) -> PW expand -> DW(3) -> PW project
    'ib'            0        3      PW expand -> DW(3) -> PW project   (MobileNetV2 IB)
    'ffn'           0        0      PW expand -> PW project            (Transformer FFN)

The macro-architecture follows ConvNeXt: a patchify stem, four stages separated by
"norm + 2x2 stride-2 conv" downsampling layers, global average pooling, a final
norm and a linear classifier. The normalization layer used everywhere is selectable
('bn' = BatchNorm, 'ln' = channels-first LayerNorm as in ConvNeXt, 'gn1' = GroupNorm
with a single group, i.e. LayerNorm over (C, H, W), 'none' = no normalization).
"""
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
from timm.layers import LayerNorm, LayerNorm2d, trunc_normal_
from timm.models._efficientnet_blocks import UniversalInvertedResidual

UIB_VARIANTS = {
    # name: (dw_kernel_size_start, dw_kernel_size_mid); None -> use the block kernel size
    'convnext': (None, 0),
    'extra_dw': (None, 3),
    'ib': (0, 3),
    'ffn': (0, 0),
}


class GroupNorm1(nn.GroupNorm):
    """GroupNorm with one group == LayerNorm over (C, H, W) of every sample."""

    def __init__(self, num_channels, eps=1e-5, affine=True, **kwargs):
        super().__init__(1, num_channels, eps=eps, affine=affine)


def get_norm_layers(norm: str):
    """Return (2d norm for feature maps, 1d norm for the pooled head features)."""
    if norm == 'bn':
        return nn.BatchNorm2d, nn.BatchNorm1d
    if norm == 'ln':
        return partial(LayerNorm2d, eps=1e-6), partial(LayerNorm, eps=1e-6)
    if norm == 'gn1':
        return GroupNorm1, partial(LayerNorm, eps=1e-6)
    if norm == 'none':
        return (lambda c, **kw: nn.Identity()), (lambda c, **kw: nn.Identity())
    raise ValueError(f'unknown norm {norm}')


class ConvNeXtUIB(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            in_chans: int = 3,
            depths: Sequence[int] = (2, 2, 6, 2),
            dims: Sequence[int] = (48, 96, 192, 384),
            stem_stride: int = 1,
            block: str = 'convnext',
            kernel_size: int = 7,
            exp_ratio: float = 4.0,
            norm: str = 'bn',
            act_layer=nn.GELU,
            drop_path_rate: float = 0.1,
            layer_scale_init_value: float = 1e-6,
            head_dropout: float = 0.0,
    ):
        super().__init__()
        assert len(depths) == len(dims)
        self.norm_name = norm
        norm2d, norm1d = get_norm_layers(norm)
        dw_start, dw_mid = UIB_VARIANTS[block]
        dw_start = kernel_size if dw_start is None else dw_start

        # ---- stem: patchify (stride s) conv + norm, like ConvNeXt (4x4/4 on ImageNet) ----
        k = max(stem_stride, 3) if stem_stride == 1 else stem_stride
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=k, stride=stem_stride, padding=(k - stem_stride + 1) // 2 if stem_stride == 1 else 0),
            norm2d(dims[0]),
        )

        # stochastic depth increases linearly with depth (as in ConvNeXt)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        prev = dims[0]
        for i, (d, c) in enumerate(zip(depths, dims)):
            layers = []
            if i > 0:  # ConvNeXt-style downsampling layer: norm -> 2x2 conv, stride 2
                layers += [norm2d(prev), nn.Conv2d(prev, c, kernel_size=2, stride=2)]
            for j in range(d):
                layers.append(UniversalInvertedResidual(
                    c, c,
                    dw_kernel_size_start=dw_start,
                    dw_kernel_size_mid=dw_mid,
                    exp_ratio=exp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm2d,
                    drop_path_rate=dpr[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                ))
            cur += d
            prev = c
            self.stages.append(nn.Sequential(*layers))

        self.head_norm = norm1d(dims[-1])
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        return x

    def forward_head(self, x):
        x = x.mean(dim=(2, 3))  # global average pooling
        x = self.head_norm(x)
        return self.head(self.head_drop(x))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


MODEL_CONFIGS = {
    # ~1.0M params, 16x16 after stem
    'tiny': dict(depths=(2, 2, 4, 2), dims=(32, 64, 128, 256), stem_stride=2),
    # ~3.6M params
    'small': dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), stem_stride=2),
}


def create_model(name: str = 'tiny', **kwargs) -> ConvNeXtUIB:
    cfg = dict(MODEL_CONFIGS[name])
    cfg.update(kwargs)
    return ConvNeXtUIB(**cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
