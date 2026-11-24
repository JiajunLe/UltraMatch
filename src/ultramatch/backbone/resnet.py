import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=bias,
    )


class RepConvBN(nn.Module):
    """Training-time multi-branch convolution fused to one 3x3 at inference."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        use_1x1=True,
        use_directional=True,
        shared_bn=False,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("RepConvBN channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = False
        self.shared_bn = shared_bn

        branch_3x3 = conv3x3(
            in_channels,
            out_channels,
            stride=stride,
            groups=groups,
        )
        branch_1x1 = (
            conv1x1(
                in_channels,
                out_channels,
                stride=stride,
                groups=groups,
            )
            if use_1x1
            else None
        )
        branch_1x3 = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 3),
                stride=stride,
                padding=(0, 1),
                groups=groups,
                bias=False,
            )
            if use_directional
            else None
        )
        branch_3x1 = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 1),
                stride=stride,
                padding=(1, 0),
                groups=groups,
                bias=False,
            )
            if use_directional
            else None
        )
        if shared_bn:
            self.rbr_3x3 = branch_3x3
            self.rbr_1x1 = branch_1x1
            self.rbr_1x3 = branch_1x3
            self.rbr_3x1 = branch_3x1
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.rbr_3x3 = self._conv_bn(branch_3x3, out_channels)
            self.rbr_1x1 = self._conv_bn(branch_1x1, out_channels)
            self.rbr_1x3 = self._conv_bn(branch_1x3, out_channels)
            self.rbr_3x1 = self._conv_bn(branch_3x1, out_channels)

    @staticmethod
    def _conv_bn(conv, out_channels):
        if conv is None:
            return None
        return nn.Sequential(conv, nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.deploy:
            return self.reparam(x)

        output = self.rbr_3x3(x)
        for branch in (self.rbr_1x1, self.rbr_1x3, self.rbr_3x1):
            if branch is not None:
                output = output + branch(x)
        if self.shared_bn:
            output = self.bn(output)
        return output

    @staticmethod
    def _fuse_kernel_bn(kernel, conv_bias, bn):
        conv_bias = (
            conv_bias
            if conv_bias is not None
            else torch.zeros(
                kernel.shape[0],
                device=kernel.device,
                dtype=kernel.dtype,
            )
        )
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = bn.bias + (conv_bias - bn.running_mean) * scale
        return fused_kernel, fused_bias

    @classmethod
    def _fuse_conv_bn(cls, branch):
        conv, bn = branch[0], branch[1]
        return cls._fuse_kernel_bn(conv.weight, conv.bias, bn)

    @staticmethod
    def _pad_to_3x3(kernel):
        height, width = kernel.shape[-2:]
        if (height, width) == (3, 3):
            return kernel
        if height > 3 or width > 3:
            raise ValueError(
                f"Cannot pad a {height}x{width} kernel to 3x3"
            )
        pad_height, pad_width = 3 - height, 3 - width
        return F.pad(
            kernel,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return self.reparam.weight, self.reparam.bias

        if self.shared_bn:
            kernel = self.rbr_3x3.weight
            for branch in (self.rbr_1x1, self.rbr_1x3, self.rbr_3x1):
                if branch is not None:
                    kernel = kernel + self._pad_to_3x3(branch.weight)
            return self._fuse_kernel_bn(kernel, None, self.bn)

        kernel, bias = self._fuse_conv_bn(self.rbr_3x3)
        for branch in (self.rbr_1x1, self.rbr_1x3, self.rbr_3x1):
            if branch is None:
                continue
            branch_kernel, branch_bias = self._fuse_conv_bn(branch)
            kernel = kernel + self._pad_to_3x3(branch_kernel)
            bias = bias + branch_bias
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return self
        if self.training:
            raise RuntimeError("RepConvBN fusion requires eval mode")

        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.copy_(kernel)
        self.reparam.bias.copy_(bias)
        self.reparam.train(self.training)

        del self.rbr_3x3
        del self.rbr_1x1
        del self.rbr_1x3
        del self.rbr_3x1
        if self.shared_bn:
            del self.bn
        self.deploy = True
        return self

    @torch.no_grad()
    def set_auxiliary_bn_scale(self, value=0.0):
        if self.deploy:
            return self
        for branch in (self.rbr_1x1, self.rbr_1x3, self.rbr_3x1):
            if branch is not None:
                if self.shared_bn:
                    branch.weight.mul_(value)
                else:
                    nn.init.constant_(branch[1].weight, value)
                    nn.init.zeros_(branch[1].bias)
        return self


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        if not self.training and not torch.is_grad_enabled():
            # Both branches are dead after this residual. Reuse ``y`` during
            # inference instead of allocating another high-resolution tensor.
            return self.relu(y.add_(x))
        return self.relu(x + y)


class RepBasicBlock(nn.Module):
    """Residual block whose two convolutions are training-reparameterized."""

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        use_1x1=True,
        use_directional=True,
        shared_bn=False,
    ):
        super().__init__()
        self.conv1 = RepConvBN(
            in_planes,
            planes,
            stride=stride,
            use_1x1=use_1x1,
            use_directional=use_directional,
            shared_bn=shared_bn,
        )
        self.conv2 = RepConvBN(
            planes,
            planes,
            use_1x1=use_1x1,
            use_directional=use_directional,
            shared_bn=shared_bn,
        )
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if not self.training and not torch.is_grad_enabled():
            return self.relu(output.add_(residual))
        return self.relu(output + residual)


class ResNet18(nn.Module):
    """
    Fewer channels
    """

    def __init__(self, config=None):
        super().__init__()
        # Config
        backbone_config = config["backbone"]
        block_dims = backbone_config["block_dims"]
        self.rep_enabled = backbone_config.get("rep_enabled", False)
        self.rep_stages = tuple(
            int(stage) for stage in backbone_config.get("rep_stages", [])
        )
        self.rep_fine = backbone_config.get("rep_fine", False)
        self.rep_use_1x1 = backbone_config.get("rep_use_1x1", True)
        self.rep_use_directional = backbone_config.get(
            "rep_use_directional", True
        )
        self.rep_shared_bn = backbone_config.get("rep_shared_bn", False)
        self.rep_auxiliary_bn_init = float(
            backbone_config.get("rep_auxiliary_bn_init", 0.0)
        )
        invalid_stages = set(self.rep_stages) - {1, 2, 3, 4, 5}
        if invalid_stages:
            raise ValueError(
                f"BACKBONE.REP_STAGES contains invalid stages: "
                f"{sorted(invalid_stages)}"
            )
        if not self.rep_enabled and (self.rep_stages or self.rep_fine):
            raise ValueError(
                "BACKBONE.REP_ENABLED is required by REP_STAGES/REP_FINE"
            )

        # Networks
        self.conv1 = nn.Conv2d(
            1, block_dims[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(block_dims[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_stage(
            1, block_dims[0], block_dims[0], stride=1
        )  # 1/2
        self.layer2 = self._make_stage(
            2, block_dims[0], block_dims[1], stride=2
        )  # 1/4
        self.layer3 = self._make_stage(
            3, block_dims[1], block_dims[2], stride=2
        )  # 1/8
        self.layer4 = self._make_stage(
            4, block_dims[2], block_dims[3], stride=2
        )  # 1/16
        self.layer5 = self._make_stage(
            5, block_dims[3], block_dims[4], stride=2
        )  # 1/32

        # For fine matching
        fine_block = RepBasicBlock if self.rep_fine else BasicBlock
        self.fine_conv = nn.Sequential(
            self._make_layer(
                fine_block,
                block_dims[2],
                block_dims[2],
                stride=1,
                **self._rep_kwargs(fine_block),
            ),
            conv1x1(block_dims[2], block_dims[4]),
            nn.BatchNorm2d(block_dims[4]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for module in self.modules():
            if isinstance(module, RepConvBN):
                module.set_auxiliary_bn_scale(
                    self.rep_auxiliary_bn_init
                )

    def _rep_kwargs(self, block):
        if block is not RepBasicBlock:
            return {}
        return {
            "use_1x1": self.rep_use_1x1,
            "use_directional": self.rep_use_directional,
            "shared_bn": self.rep_shared_bn,
        }

    def _make_stage(self, stage, in_dim, out_dim, stride):
        block = (
            RepBasicBlock
            if self.rep_enabled and stage in self.rep_stages
            else BasicBlock
        )
        return self._make_layer(
            block,
            in_dim,
            out_dim,
            stride=stride,
            **self._rep_kwargs(block),
        )

    def _make_layer(self, block, in_dim, out_dim, stride=1, **kwargs):
        layer1 = block(in_dim, out_dim, stride=stride, **kwargs)
        layer2 = block(out_dim, out_dim, stride=1, **kwargs)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16
        x5 = self.layer5(x4)  # 1/32

        xf = self.fine_conv(x3)  # 1/8

        return [x3, x4, x5, xf]

    def forward_route(self, x):
        """Return only the 1/32 feature needed by router-only training."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError("Backbone RepConvBN fusion requires eval mode")
        rep_modules = [
            module
            for module in self.modules()
            if isinstance(module, RepConvBN)
        ]
        for module in rep_modules:
            module.switch_to_deploy()
        return self
