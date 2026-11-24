import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model

    `Density estimation using Real NVP
    arXiv: <https://arxiv.org/abs/1605.08803>`_.

    Code is modified from `the mmpose implementation of RLE
    <https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py>`_.
    """

    @staticmethod
    def get_scale_net(channel):
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, channel),
            nn.GELU(),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, 2),
            nn.Tanh(),
        )

    @staticmethod
    def get_trans_net(channel):
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, channel),
            nn.GELU(),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, 2),
        )

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self, channel=64):
        super(RealNVP, self).__init__()
        self.channel = channel
        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer(
            "mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32)
        )

        self.s = torch.nn.ModuleList(
            [self.get_scale_net(self.channel) for _ in range(len(self.mask))]
        )
        self.t = torch.nn.ModuleList(
            [self.get_trans_net(self.channel) for _ in range(len(self.mask))]
        )
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""

        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""

        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


def soft_argmax(x, temperature=1.0):
    L = x.shape[1]
    assert L % 2  # L is odd to ensure symmetry
    idx = torch.arange(0, L, 1, device=x.device)[None]
    scale_x = x / temperature
    out = F.softmax(scale_x, dim=1) * idx
    out = out.sum(dim=1, keepdim=True)

    return out


class TinyFineMatching(nn.Module):
    """Shared Parameter Tiny Fine Matching for bidirectional subpixel refinement."""

    matcher_type = "tiny_fine_matching"

    def __init__(self, config):
        super().__init__()
        fine = config["fine"]
        matcher_type = fine.get("matcher_type", self.matcher_type)
        if matcher_type != self.matcher_type:
            raise ValueError(
                f"Unsupported fine matcher {matcher_type!r}. Expected {self.matcher_type!r}."
            )

        self.local_resolution = config["local_resolution"]
        self.coord_length = fine["coord_length"]
        self.sigma_selection = fine["sigma_selection"]
        self.sigma_thr = fine["sigma_thr"]
        self.mconf_thr = config["coarse"]["mconf_thr"]
        self.border_rm = config["coarse"]["border_rm"] * self.local_resolution
        self.deploy = config["deploy"]
        self.input_dim = int(config["neck"]["d_model"])
        self.descriptor_hidden_dim = int(fine.get("descriptor_hidden_dim", 256))
        self.pair_hidden_dim = int(fine.get("pair_hidden_dim", 512))
        self.axis_hidden_dim = int(fine.get("axis_hidden_dim", 256))
        if self.input_dim != 256:
            raise ValueError("TinyFineMatching requires 256-D 1/8 CIM features")
        if self.coord_length != 16:
            raise ValueError("TinyFineMatching requires exactly 17 axis logits")
        if not fine["bi_directional_refine"]:
            raise ValueError("TinyFineMatching requires bidirectional refinement")
        if min(
            self.descriptor_hidden_dim,
            self.pair_hidden_dim,
            self.axis_hidden_dim,
        ) < 1:
            raise ValueError("TinyFineMatching dimensions must be positive")

        self.flow = RealNVP()
        self.descriptor_norm = nn.LayerNorm(self.input_dim)
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.descriptor_hidden_dim),
            nn.GELU(),
            nn.Linear(self.descriptor_hidden_dim, self.input_dim),
        )
        pair_dim = 2 * self.input_dim
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_dim, self.pair_hidden_dim),
            nn.GELU(),
            nn.Linear(self.pair_hidden_dim, pair_dim),
        )
        self.context_norm = nn.LayerNorm(pair_dim)
        axis_output_dim = self.coord_length + 2
        self.x_axis_head = nn.Sequential(
            nn.Linear(pair_dim, self.axis_hidden_dim),
            nn.GELU(),
            nn.Linear(self.axis_hidden_dim, axis_output_dim),
        )
        self.y_axis_head = nn.Sequential(
            nn.Linear(pair_dim, self.axis_hidden_dim),
            nn.GELU(),
            nn.Linear(self.axis_hidden_dim, axis_output_dim),
        )
        self.reset_prediction_parameters()

    def reset_prediction_parameters(self):
        containers = (
            self.descriptor_encoder,
            self.pair_projection,
            self.x_axis_head,
            self.y_axis_head,
        )
        for container in containers:
            for module in container.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        for head in (self.x_axis_head, self.y_axis_head):
            nn.init.xavier_uniform_(head[-1].weight, gain=0.1)
            nn.init.zeros_(head[-1].bias)

    def _pair_context(self, query, reference):
        query = query + self.descriptor_encoder(self.descriptor_norm(query))
        reference = reference + self.descriptor_encoder(
            self.descriptor_norm(reference)
        )
        pair = torch.cat((query, reference), dim=-1)
        context = pair + self.pair_projection(self.pair_norm(pair))
        return self.context_norm(context)

    @staticmethod
    def _flatten_directions(prediction):
        prediction01, prediction10 = prediction.chunk(2, dim=1)
        return torch.cat(
            (
                prediction01.reshape(-1, prediction.shape[-1]),
                prediction10.reshape(-1, prediction.shape[-1]),
            ),
            dim=0,
        )

    def _publish_rle_outputs(self, coord, sigma, data):
        if data.get("target_uv") is not None:
            gt_uv = data["target_uv"]
            mask = data["target_uv_weight"].clone()

            if mask.sum() == 0:
                if self.training:
                    # Keep one finite target in the DDP graph. UltraMatchLoss
                    # gives the all-invalid batch zero fine-loss weight.
                    mask[0] = True
                    data["target_uv"] = torch.nan_to_num(gt_uv)
                else:
                    mask_coord = coord[:0]
                    mask_sigma = sigma[:0]
                    nf_loss = coord[:0]

            if mask.any():
                mask_coord = torch.nan_to_num(coord[mask])
                mask_gt_uv = torch.nan_to_num(gt_uv[mask])
                mask_sigma = torch.nan_to_num(
                    sigma[mask], nan=0.5, posinf=1.0, neginf=0.0
                ).clamp(1e-6, 1.0 - 1e-6)
                normalized_error = (mask_coord - mask_gt_uv) / mask_sigma
                log_phi = self.flow.log_prob(normalized_error).unsqueeze(-1)
                nf_loss = torch.log(mask_sigma) - log_phi

            data.update(
                {
                    "pred_coord": coord,
                    "pred_score": 1.0 - sigma.mean(dim=-1).flatten(),
                    "mask_coord": mask_coord,
                    "mask_sigma": mask_sigma,
                    "nf_loss": nf_loss,
                }
            )
        else:
            data.update(
                {
                    "pred_coord": coord,
                    "pred_score": 1.0 - sigma.mean(dim=-1).flatten(),
                }
            )

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data=None):
        query = feat_f0 + feat_c0
        reference = feat_f1 + feat_c1
        return self.forward_precombined(query, reference, data)

    def forward_precombined(self, query, reference, data=None):
        if data is None:
            data = {}
        if query.shape != reference.shape or query.shape[-1] != self.input_dim:
            raise ValueError(
                "TinyFineMatching expects matching 256-D query/reference rows"
            )
        if query.shape[1] % 2:
            raise ValueError(
                "TinyFineMatching expects equal bidirectional row counts"
            )

        # The RealNVP likelihood is sensitive to reduced-precision sigma values.
        with torch.autocast(device_type=query.device.type, enabled=False):
            context = self._pair_context(query.float(), reference.float())
            x_prediction = self._flatten_directions(
                self.x_axis_head(context)
            )
            y_prediction = self._flatten_directions(
                self.y_axis_head(context)
            )
            x_logits = x_prediction[:, : self.coord_length + 1]
            y_logits = y_prediction[:, : self.coord_length + 1]
            coord_x = soft_argmax(x_logits) / self.coord_length - 0.5
            coord_y = soft_argmax(y_logits) / self.coord_length - 0.5
            sigma_x = x_prediction[:, -1:].sigmoid()
            sigma_y = y_prediction[:, -1:].sigmoid()
            coord = torch.cat((coord_x, coord_y), dim=-1)
            sigma = torch.cat((sigma_x, sigma_y), dim=-1)
            self._publish_rle_outputs(coord, sigma, data)

        if not self.deploy:
            self.final_matching_selection(data)
        return data["pred_coord"], data["pred_score"]

    @torch.no_grad()
    def final_matching_selection(self, data):
        offset = data["pred_coord"] * self.local_resolution

        fine_offset01, fine_offset10 = offset.clamp(
            -self.local_resolution / 2, self.local_resolution / 2
        ).chunk(2)

        h0, w0 = data["hw0_i"]
        h1, w1 = data["hw1_i"]
        scale0 = data["scale0"][data["b_ids"]] if "scale0" in data else 1.0
        scale1 = data["scale1"][data["b_ids"]] if "scale1" in data else 1.0
        scale0_w = scale0[:, 0] if "scale0" in data else 1.0
        scale0_h = scale0[:, 1] if "scale0" in data else 1.0
        scale1_w = scale1[:, 0] if "scale1" in data else 1.0
        scale1_h = scale1[:, 1] if "scale1" in data else 1.0

        # Filter by mconf and border
        mkpts0_f = data["mkpts0_c"]
        mkpts1_f = data["mkpts1_c"] + fine_offset01 * scale1
        coarse_conf_mask = data.get("fine_candidate_mask")
        if coarse_conf_mask is None:
            coarse_conf_mask = data["mconf"] > self.mconf_thr
        mask = (
            coarse_conf_mask
            & (mkpts0_f[:, 0] >= self.border_rm)
            & (mkpts0_f[:, 0] <= w0 * scale0_w - self.border_rm)
            & (mkpts0_f[:, 1] >= self.border_rm)
            & (mkpts0_f[:, 1] <= h0 * scale0_h - self.border_rm)
            & (mkpts1_f[:, 0] >= self.border_rm)
            & (mkpts1_f[:, 0] <= w1 * scale1_w - self.border_rm)
            & (mkpts1_f[:, 1] >= self.border_rm)
            & (mkpts1_f[:, 1] <= h1 * scale1_h - self.border_rm)
        )
        mkpts0_f_ = data["mkpts0_c"] + fine_offset10 * scale0
        mkpts1_f_ = data["mkpts1_c"]
        mask_ = (
            coarse_conf_mask
            & (mkpts0_f_[:, 0] >= self.border_rm)
            & (mkpts0_f_[:, 0] <= w0 * scale0_w - self.border_rm)
            & (mkpts0_f_[:, 1] >= self.border_rm)
            & (mkpts0_f_[:, 1] <= h0 * scale0_h - self.border_rm)
            & (mkpts1_f_[:, 0] >= self.border_rm)
            & (mkpts1_f_[:, 0] <= w1 * scale1_w - self.border_rm)
            & (mkpts1_f_[:, 1] >= self.border_rm)
            & (mkpts1_f_[:, 1] <= h1 * scale1_h - self.border_rm)
        )

        mkpts0_f = torch.cat([mkpts0_f, mkpts0_f_])
        mkpts1_f = torch.cat([mkpts1_f, mkpts1_f_])
        mask = torch.cat([mask, mask_])
        data["mconf"] = torch.cat([data["mconf"], data["mconf"]])
        data["b_ids"] = torch.cat([data["b_ids"], data["b_ids"]])

        if self.sigma_selection:
            # Keep the more confident direction for each pair.
            pred_score01, pred_score10 = data["pred_score"].chunk(2)
            pred_score_mask = pred_score01 > pred_score10
            pred_score_mask = torch.cat([pred_score_mask, ~pred_score_mask])
            pred_score_mask &= data["pred_score"] > self.sigma_thr
            mask &= pred_score_mask

        data.update(
            {
                "m_bids": data["b_ids"][mask],
                "mkpts0_f": mkpts0_f[mask],
                "mkpts1_f": mkpts1_f[mask],
                "mconf": data["mconf"][mask],
            }
        )


FineMatching = TinyFineMatching
