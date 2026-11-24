import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


_CONV_TYPES = (nn.Conv1d, nn.Conv2d)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)


def _fuse_pair(parent, conv_name, bn_name):
    conv = getattr(parent, conv_name, None)
    bn = getattr(parent, bn_name, None)
    if not isinstance(conv, _CONV_TYPES) or not isinstance(bn, _BN_TYPES):
        return False
    if conv.training or bn.training:
        raise RuntimeError("Conv-BN reparameterization requires eval mode")
    setattr(parent, conv_name, fuse_conv_bn_eval(conv, bn))
    setattr(parent, bn_name, nn.Identity())
    return True


def reparameterize_conv_bn(module):
    """Fold every supported inference Conv-BN pair in-place.

    The training/checkpoint graph remains unchanged until this function is
    called on an eval model. The operation is idempotent.
    """
    if module.training:
        raise RuntimeError("Conv-BN reparameterization requires eval mode")

    for child in list(module.children()):
        reparameterize_conv_bn(child)

    # ResNet-style explicitly named pairs.
    for suffix in ("", "1", "2", "3"):
        conv_name = "conv" + suffix
        bn_name = "bn" + suffix
        _fuse_pair(module, conv_name, bn_name)

    # Conv{1,2}d_BN_Act and other sequential Conv-BN containers.
    if isinstance(module, nn.Sequential):
        names = list(module._modules)
        for conv_name, bn_name in zip(names, names[1:]):
            _fuse_pair(module, conv_name, bn_name)

    return module
