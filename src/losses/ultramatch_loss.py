from loguru import logger

import torch
import torch.nn as nn
import math


class UltraMatchLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config["ultramatch"]["loss"]
        fine_type = self.loss_config.get("fine_type", "rle")
        if fine_type != "rle":
            raise ValueError(f"Unknown fine loss: {fine_type}")
        self.sparse_spvs = self.loss_config["sparse_spvs"]
        self.router_config = config["ultramatch"].get("router", {})
        self.router_enabled = self.router_config.get("enabled", False)
        self.router_train_only = self.router_config.get("train_only", False)
        self.router_sparse_only = self.router_config.get("sparse_only", False)
        self.router_weight = self.router_config.get("loss_weight", 1.0)
        self.router_coverage_weight = self.router_config.get(
            "coverage_loss_weight", 0.0
        )
        self.router_rank_weight = self.router_config.get(
            "rank_loss_weight", 0.0
        )
        self.routed_coarse_weight = self.router_config.get(
            "routed_coarse_weight", 0.0
        )

        # coarse-level
        self.c_pos_w = self.loss_config["pos_weight"]
        self.c_neg_w = self.loss_config["neg_weight"]
        # fine-level
        self.q_distribution = self.loss_config["q_distribution"]

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # Keep an empty class from producing NaN in its mean loss.
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_neg_w = 0.0

        if self.loss_config["coarse_type"] == "cross_entropy":
            assert (
                not self.sparse_spvs
            ), "Sparse Supervision for cross-entropy not implemented!"
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = -torch.log(conf[pos_mask])
            loss_neg = -torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

        elif self.loss_config["coarse_type"] == "focal":
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config["focal_alpha"]
            gamma = self.loss_config["focal_gamma"]

            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = -alpha * \
                    torch.pow(1 - pos_conf, gamma) * pos_conf.log()

                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]

                loss = c_pos_w * loss_pos.mean()
                return loss
            else:
                loss_pos = (
                    -alpha
                    * torch.pow(1 - conf[pos_mask], gamma)
                    * (conf[pos_mask]).log()
                )

                loss_neg = (
                    -alpha
                    * torch.pow(conf[neg_mask], gamma)
                    * (1 - conf[neg_mask]).log()
                )
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]

                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        else:
            raise ValueError(
                "Unknown coarse loss: {type}".format(
                    type=self.loss_config["coarse_type"]
                )
            )

    def compute_sparse_coarse_loss(self, conf, data):
        """Focal positive loss without materializing a dense GT matrix."""
        if self.loss_config["coarse_type"] != "focal":
            raise ValueError("Sparse coarse supervision requires focal loss")

        valid = data.get(
            "spv_gt_mask",
            torch.ones_like(data["spv_b_ids"], dtype=torch.bool),
        ).bool()
        if not valid.any():
            return conf.reshape(-1)[0] * 0.0

        b_ids = data["spv_b_ids"][valid]
        i_ids = data["spv_i_ids"][valid]
        j_ids = data["spv_j_ids"][valid]
        pos_conf = conf[b_ids, i_ids, j_ids].clamp(1e-6, 1 - 1e-6)
        alpha = self.loss_config["focal_alpha"]
        gamma = self.loss_config["focal_gamma"]
        loss_pos = -alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()

        if "mask0" in data:
            weight = (
                data["mask0"].flatten(1)[b_ids, i_ids]
                * data["mask1"].flatten(1)[b_ids, j_ids]
            ).to(loss_pos.dtype)
            loss_pos = loss_pos * weight
        return self.c_pos_w * loss_pos.mean()

    def compute_routed_coarse_loss(self, confidence):
        if confidence.numel() == 0:
            return confidence.sum() * 0.0
        confidence = confidence.clamp(1e-6, 1 - 1e-6)
        alpha = self.loss_config["focal_alpha"]
        gamma = self.loss_config["focal_gamma"]
        return (
            -alpha
            * torch.pow(1 - confidence, gamma)
            * confidence.log()
        ).mean()

    def logQ(self, gt_uv, pred_jts, sigma):
        assert self.q_distribution in ["laplace", "gaussian"]

        error = (pred_jts - gt_uv) / (sigma + 1e-9)

        if self.q_distribution == "laplace":
            loss_q = torch.log(sigma * 2) + torch.abs(error)
        else:
            loss_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

        return loss_q

    def compute_rle_loss(self, data, f_weight=1):
        gt_uv = data["target_uv"]
        gt_uv_weight = data["target_uv_weight"]

        if gt_uv_weight.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning(
                    "assign a false supervision to avoid ddp deadlock")
                gt_uv_weight[0] = True
                f_weight = 0.0
            else:
                return None

        residual = True
        if residual:
            Q_logprob = self.logQ(
                gt_uv[gt_uv_weight], data["mask_coord"], data["mask_sigma"]
            )
            loss = Q_logprob + data["nf_loss"]

        return loss.mean() * f_weight

    @torch.no_grad()
    def compute_c_weight(self, data):
        """compute element-wise weights for computing coarse-level loss."""
        if "mask0" in data:
            c_weight = (
                data["mask0"].flatten(-2)[..., None]
                * data["mask1"].flatten(-2)[:, None]
            ).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        if self.router_train_only:
            loss_router = data["loss_router"]
            loss = self.router_weight * loss_router
            if self.router_coverage_weight > 0:
                loss_coverage = data["loss_router_coverage"]
                loss = loss + self.router_coverage_weight * loss_coverage
                loss_scalars["loss_router_coverage"] = (
                    loss_coverage.detach().cpu()
                )
            if self.router_rank_weight > 0:
                loss_rank = data["loss_router_rank"]
                loss = loss + self.router_rank_weight * loss_rank
                loss_scalars["loss_router_rank"] = loss_rank.detach().cpu()
            loss_scalars.update(
                {
                    "loss_router": loss_router.detach().cpu(),
                    "route_recall01": data["route_recall01"].cpu(),
                    "route_recall10": data["route_recall10"].cpu(),
                    "loss": loss.detach().cpu(),
                }
            )
            data.update({"loss": loss, "loss_scalars": loss_scalars})
            return

        loss = None
        if not self.router_sparse_only:
            # Dense confidence supervision uses the full O((HW)^2) matrix.
            # Sparse-only training uses routed matches and block supervision.
            c_weight = (
                None if self.sparse_spvs else self.compute_c_weight(data)
            )
            if self.sparse_spvs:
                loss_c = self.compute_sparse_coarse_loss(
                    data["conf_matrix"], data
                )
            else:
                loss_c = self.compute_coarse_loss(
                    data["conf_matrix"],
                    data["conf_matrix_gt"],
                    weight=c_weight,
                )
            loss = loss_c * self.loss_config["coarse_weight"]
            loss_scalars.update({"loss_c": loss_c.detach().cpu()})

        if self.router_enabled and "loss_router" in data:
            loss_router = data["loss_router"]
            router_term = self.router_weight * loss_router
            if self.router_coverage_weight > 0:
                loss_coverage = data["loss_router_coverage"]
                router_term = (
                    router_term
                    + self.router_coverage_weight * loss_coverage
                )
                loss_scalars["loss_router_coverage"] = (
                    loss_coverage.detach().cpu()
                )
            if self.router_rank_weight > 0:
                loss_rank = data["loss_router_rank"]
                router_term = router_term + self.router_rank_weight * loss_rank
                loss_scalars["loss_router_rank"] = loss_rank.detach().cpu()
            loss = router_term if loss is None else loss + router_term
            loss_scalars.update(
                {
                    "loss_router": loss_router.detach().cpu(),
                    "route_recall01": data["route_recall01"].cpu(),
                    "route_recall10": data["route_recall10"].cpu(),
                }
            )
            if "gt_route_injected_slots" in data:
                loss_scalars["gt_route_injected_slots"] = data[
                    "gt_route_injected_slots"
                ].detach().cpu()
            if "route_size_used" in data:
                loss_scalars["route_size_used"] = data[
                    "route_size_used"
                ].detach().cpu()

        if (
            self.routed_coarse_weight > 0
            and "routed_gt_confidence" in data
        ):
            loss_routed_c = self.compute_routed_coarse_loss(
                data["routed_gt_confidence"]
            )
            routed_term = self.routed_coarse_weight * loss_routed_c
            loss = routed_term if loss is None else loss + routed_term
            loss_scalars.update(
                {
                    "loss_routed_c": loss_routed_c.detach().cpu(),
                    "routed_gt_recall": data["routed_gt_recall"].cpu(),
                }
            )

        loss_f = self.compute_rle_loss(
            data=data, f_weight=self.loss_config["fine_weight"]
        )
        if loss_f is not None:
            loss = loss_f if loss is None else loss + loss_f
            loss_scalars.update(
                {"loss_f": min(loss_f.clone().detach().cpu(),
                               torch.tensor(1.0))}
            )
        else:
            assert self.training is False
            # Log the upper bound when validation has no fine targets.
            loss_scalars.update({"loss_f": torch.tensor(1.0)})

        if loss is None:
            raise RuntimeError("No active UltraMatch training objective")
        loss_scalars.update({"loss": loss.detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
