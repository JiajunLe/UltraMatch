import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockRouter(nn.Module):
    """Shared low-rank projection used for symmetric block routing."""

    def __init__(self, input_dim, route_dim, score_prior_init=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, route_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(route_dim),
        )
        nn.init.kaiming_normal_(
            self.proj[0].weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.ones_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)
        self.score_prior_weight = nn.Parameter(
            torch.tensor(float(score_prior_init))
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Use the configured prior weight when a checkpoint omits it.
        prior_key = prefix + "score_prior_weight"
        if prior_key not in state_dict:
            state_dict[prior_key] = self.score_prior_weight.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, feat0, feat1):
        if feat0.shape[-2:] == feat1.shape[-2:]:
            batch0 = feat0.shape[0]
            projected = self.proj(torch.cat((feat0, feat1), dim=0))
            projected0, projected1 = projected[:batch0], projected[batch0:]
        else:
            projected0 = self.proj(feat0)
            projected1 = self.proj(feat1)

        descriptors0 = F.normalize(
            projected0.flatten(2).transpose(1, 2), p=2, dim=-1
        )
        descriptors1 = F.normalize(
            projected1.flatten(2).transpose(1, 2), p=2, dim=-1
        )
        return descriptors0, descriptors1
