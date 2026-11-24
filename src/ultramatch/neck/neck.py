import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.resnet import RepConvBN
from .loftr_module.transformer import LocalFeatureTransformer


class Conv2d_BN_Act(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act=None,
        drop=None,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if act != None:
            self.add_module("a", act)
        if drop != None:
            self.add_module("d", nn.Dropout(drop))


class CIM(nn.Module):
    """Feature Aggregation, Correlation Injection Module"""

    def __init__(self, config):
        super(CIM, self).__init__()

        self.block_dims = config["backbone"]["block_dims"]
        self.drop = config["fine"]["droprate"]
        self.rep_dw = bool(config["neck"].get("rep_dw", False))

        self.fc32 = Conv2d_BN_Act(
            self.block_dims[-1], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc16 = Conv2d_BN_Act(
            self.block_dims[-2], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc8 = Conv2d_BN_Act(
            self.block_dims[-3], self.block_dims[-1], 1, drop=self.drop
        )
        self.att32 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.att16 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.dwconv16 = self._make_dwconv()
        self.dwconv8 = self._make_dwconv()

        self.loftr_32 = LocalFeatureTransformer(config["neck"])

    def _make_dwconv(self):
        channels = self.block_dims[-1]
        if self.rep_dw:
            depthwise = RepConvBN(
                channels,
                channels,
                groups=channels,
                use_1x1=True,
                use_directional=True,
                shared_bn=True,
            )
            # Initially only the 3x3 depthwise branch contributes.
            depthwise.set_auxiliary_bn_scale(0.0)
            depthwise = nn.Sequential(depthwise, nn.GELU())
        else:
            depthwise = Conv2d_BN_Act(
                channels,
                channels,
                ks=3,
                pad=1,
                groups=channels,
                act=nn.GELU(),
            )
        return nn.Sequential(
            depthwise,
            Conv2d_BN_Act(channels, channels, 1),
        )

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError("CIM RepDW fusion requires eval mode")
        rep_modules = [
            module
            for module in self.modules()
            if isinstance(module, RepConvBN)
        ]
        for module in rep_modules:
            module.switch_to_deploy()
        return self

    def forward_router(self, feat32_0, feat32_1=None, mask_c0=None, mask_c1=None):
        """Compute only the interacted 1/32 descriptors used by the router."""
        if feat32_1 is None:
            feat32 = self.fc32(feat32_0)
            feat32_0, feat32_1 = feat32.chunk(2, dim=0)
        else:
            feat32_0 = self.fc32(feat32_0)
            feat32_1 = self.fc32(feat32_1)
        return self.loftr_32(feat32_0, feat32_1, mask_c0, mask_c1)

    def _fuse_multiscale(self, feature, attention, residual):
        if not self.training and not torch.is_grad_enabled():
            return feature.mul_(attention).add_(residual)
        return feature * attention + residual

    @staticmethod
    def _interpolate(feature, **kwargs):
        # PyTorch 2.0 CUDA has no BF16 bilinear upsample kernel. Keep only this
        # channel-independent resize in FP32, then return to the feature dtype.
        if feature.is_cuda and feature.dtype == torch.bfloat16:
            return F.interpolate(feature.float(), **kwargs).to(feature.dtype)
        return F.interpolate(feature, **kwargs)

    def forward(self, ms_feats, mask_c0=None, mask_c1=None):
        if len(ms_feats) == 3:  # same image shape
            f8, f16, f32 = ms_feats
            f32 = self.fc32(f32)

            f32_0, f32_1 = f32.chunk(2, dim=0)
            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)
            f32 = torch.cat([f32_0, f32_1], dim=0)
            route_feat0, route_feat1 = f32_0, f32_1

            f32_up = self._interpolate(
                f32, scale_factor=2.0, mode="bilinear"
            )
            att32_up = self._interpolate(
                self.att32(f32), scale_factor=2.0, mode="bilinear"
            )
            f16 = self.fc16(f16)
            f16 = self.dwconv16(
                self._fuse_multiscale(f16, att32_up, f32_up)
            )
            f16_up = self._interpolate(
                f16, scale_factor=2.0, mode="bilinear"
            )
            att16_up = self._interpolate(
                self.att16(f16), scale_factor=2.0, mode="bilinear"
            )
            f8 = self.fc8(f8)
            f8 = self.dwconv8(
                self._fuse_multiscale(f8, att16_up, f16_up)
            )

            feat_c0, feat_c1 = f8.chunk(2)

        elif len(ms_feats) == 6:  # diffirent image shape
            f8_0, f16_0, f32_0, f8_1, f16_1, f32_1 = ms_feats
            f32_0 = self.fc32(f32_0)
            f32_1 = self.fc32(f32_1)

            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)
            route_feat0, route_feat1 = f32_0, f32_1

            f8, f16, f32 = f8_0, f16_0, f32_0
            f32_up = self._interpolate(
                f32, scale_factor=2.0, mode="bilinear"
            )
            att32_up = self._interpolate(
                self.att32(f32), scale_factor=2.0, mode="bilinear"
            )
            f16 = self.fc16(f16)
            f16 = self.dwconv16(
                self._fuse_multiscale(f16, att32_up, f32_up)
            )
            f16_up = self._interpolate(
                f16, scale_factor=2.0, mode="bilinear"
            )
            att16_up = self._interpolate(
                self.att16(f16), scale_factor=2.0, mode="bilinear"
            )
            f8 = self.fc8(f8)
            f8 = self.dwconv8(
                self._fuse_multiscale(f8, att16_up, f16_up)
            )
            feat_c0 = f8

            f8, f16, f32 = f8_1, f16_1, f32_1
            f32_up = self._interpolate(
                f32, scale_factor=2.0, mode="bilinear"
            )
            att32_up = self._interpolate(
                self.att32(f32), scale_factor=2.0, mode="bilinear"
            )
            f16 = self.fc16(f16)
            f16 = self.dwconv16(
                self._fuse_multiscale(f16, att32_up, f32_up)
            )
            f16_up = self._interpolate(
                f16, scale_factor=2.0, mode="bilinear"
            )
            att16_up = self._interpolate(
                self.att16(f16), scale_factor=2.0, mode="bilinear"
            )
            f8 = self.fc8(f8)
            f8 = self.dwconv8(
                self._fuse_multiscale(f8, att16_up, f16_up)
            )
            feat_c1 = f8

        return feat_c0, feat_c1, route_feat0, route_feat1
