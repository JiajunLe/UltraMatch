import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ultramatch.head.block_router import BlockRouter
from src.ultramatch.head.fused_sparse import fused_confidence_rowmax
from src.ultramatch.utils.supervision import compute_supervision_fine

INF = 1e9  # -1e4 for fp16matmul


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_size = config["local_resolution"]
        self.thr = config["coarse"]["mconf_thr"]
        self.temperature = config["coarse"]["dsmax_temperature"]
        self.ds_opt = config["coarse"]["ds_opt"]
        self.pad_num = config["coarse"]["train_pad_num"]
        self.topk = config["coarse"]["topk"]
        self.deploy = config["deploy"]
        self.mode = config["coarse"].get("mode", "full")
        if self.mode not in {"full", "block_routed", "block_sparse"}:
            raise ValueError(f"Unknown coarse matching mode: {self.mode}")
        self.block_size = config["coarse"].get("block_size", 4)
        self.route_topk = config["coarse"].get("route_topk", 4)
        self.block_halo = config["coarse"].get("block_halo", 0)
        self.fused_sparse_selection = config["coarse"].get(
            "fused_sparse_selection", False
        )
        router_config = config.get("router", {})
        self.router_enabled = router_config.get("enabled", False)
        self.router_train_only = router_config.get("train_only", False)
        self.router_eval_routed = router_config.get("eval_routed", False)
        self.router_train_routed = router_config.get("train_routed", False)
        self.router_sparse_only = router_config.get("sparse_only", False)
        self.router_score_prior = router_config.get(
            "score_prior_enabled", False
        )
        self.router_score_prior_max = float(
            router_config.get("score_prior_max", 1.0)
        )
        self.gt_inject_epochs = float(
            router_config.get("gt_inject_epochs", 0.0)
        )
        self.train_route_sizes = tuple(
            sorted(set(int(k) for k in router_config.get("train_sizes", [])))
        )
        if any(k <= 0 for k in self.train_route_sizes):
            raise ValueError("ROUTER.TRAIN_SIZES must contain positive integers")
        self.router_rank_margin = float(
            router_config.get("rank_margin", 0.5)
        )
        self.router_temperature = router_config.get("temperature", 0.1)
        self.block_chunk_size = router_config.get("block_chunk_size", 128)
        self.router = (
            BlockRouter(
                config["neck"]["d_model"],
                router_config.get("dim", 64),
                router_config.get("score_prior_init", 0.0),
            )
            if self.router_enabled
            else None
        )
        self._layout_cache = {}

    def _apply(self, fn):
        # These layouts are device-specific and deliberately stay out of the
        # checkpoint state. Rebuild them lazily after a module move.
        self._layout_cache.clear()
        return super()._apply(fn)

    @staticmethod
    def _deterministic_index_groups(indices, output_size):
        """Build stable grouped-index metadata reusable by multiple reductions."""
        if indices.ndim != 2:
            raise ValueError("indices must have shape [B, N]")
        batch = indices.shape[0]
        batch_offsets = (
            torch.arange(batch, device=indices.device, dtype=indices.dtype)
            * output_size
        )[:, None]
        flat_indices = (indices + batch_offsets).reshape(-1)
        sorted_indices, order = torch.sort(flat_indices, stable=True)
        unique_indices, counts = torch.unique_consecutive(
            sorted_indices, return_counts=True
        )
        return order, unique_indices, counts

    @staticmethod
    def _deterministic_index_reduce(
        values, indices, output_size, reduce, groups=None
    ):
        """Reduce [batch, items] values by index without CUDA atomics.

        ``scatter_reduce`` and ``scatter_add`` are fast but nondeterministic on
        CUDA. Stable sorting followed by a segmented reduction preserves a
        fixed accumulation order and is used only when PyTorch deterministic
        algorithms are explicitly enabled for evaluation.
        """
        if values.ndim != 2 or indices.shape != values.shape:
            raise ValueError("values and indices must have matching [B, N] shapes")
        if reduce not in ("max", "sum"):
            raise ValueError(f"Unsupported deterministic reduction: {reduce}")

        batch = values.shape[0]
        flat_values = values.reshape(-1)
        if groups is None:
            groups = CoarseMatching._deterministic_index_groups(
                indices, output_size
            )
        order, unique_indices, counts = groups
        sorted_values = flat_values[order]
        reduced = torch.segment_reduce(
            sorted_values, reduce=reduce, lengths=counts
        )
        fill_value = -torch.inf if reduce == "max" else 0.0
        output = torch.full(
            (batch * output_size,),
            fill_value,
            dtype=values.dtype,
            device=values.device,
        )
        output.index_copy_(0, unique_indices, reduced)
        return output.view(batch, output_size)

    def forward(
        self,
        feat_c0,
        feat_c1,
        data,
        feat_r0=None,
        feat_r1=None,
        mask_c0=None,
        mask_c1=None,
    ):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        route_scores = None
        if self.router_enabled and data.get("spv_b_ids") is not None:
            route_scores = self.forward_router_supervision(
                feat_r0,
                feat_r1,
                data,
                mask_c0=mask_c0,
                mask_c1=mask_c1,
            )

        # Sparse-only joint training uses the same fixed-size matching graph
        # as inference. Router supervision above is constructed directly from
        # GT block pairs and does not require a dense 1/8 confidence matrix.
        sparse_only_training = (
            self.training
            and self.router_train_routed
            and self.router_sparse_only
        )
        use_block_routing = (
            self.mode in ("block_routed", "block_sparse")
            and (
                sparse_only_training
                or (
                    not self.training
                    and (
                        self.router_sparse_only
                        or data.get("spv_b_ids", None) is None
                        or self.router_train_only
                    )
                )
            )
        )
        if use_block_routing:
            if feat_r0 is None or feat_r1 is None:
                raise ValueError("block_routed coarse matching requires 1/32 route features")
            return self._forward_block_routed(
                feat_c0,
                feat_c1,
                feat_r0,
                feat_r1,
                data,
                mask_c0=mask_c0,
                mask_c1=mask_c1,
                route_scores=route_scores,
            )

        routed_feat_c0, routed_feat_c1 = feat_c0, feat_c1

        # normalize
        feat_c0, feat_c1 = map(
            lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1]
        )

        with torch.autocast(enabled=False, device_type="cuda"):
            sim_matrix = (
                torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) /
                self.temperature
            )
            del feat_c0, feat_c1
            if mask_c0 is not None:
                sim_matrix = sim_matrix.float().masked_fill(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF,
                )

        if not self.training and self.ds_opt:
            # Alternative implementation of daul-softmax operator for efficient inference
            sim_matrix = torch.exp(sim_matrix)
            conf_matrix = F.normalize(sim_matrix, p=1, dim=1) * F.normalize(
                sim_matrix, p=1, dim=2
            )
        else:
            # Native daul-softmax operator
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update(
            {
                "conf_matrix": conf_matrix,
            }
        )

        if not self.deploy:
            if self.router_train_routed and self.training:
                self._forward_block_routed(
                    routed_feat_c0,
                    routed_feat_c1,
                    feat_r0,
                    feat_r1,
                    data,
                    mask_c0=mask_c0,
                    mask_c1=mask_c1,
                )
            elif (
                self.router_eval_routed
                and self.mode in ("block_routed", "block_sparse")
                and not self.training
                and data.get("spv_b_ids", None) is not None
            ):
                self._forward_block_routed(
                    routed_feat_c0,
                    routed_feat_c1,
                    feat_r0,
                    feat_r1,
                    data,
                    mask_c0=mask_c0,
                    mask_c1=mask_c1,
                )
            else:
                # predict coarse matches from conf_matrix
                self.coarse_matching_selection(data)

        return conf_matrix # Returning the sim_matrix can be faster, but it may reduce the accuracy.

    @staticmethod
    def _to_blocks(feat, height, width, block_size):
        """Pack a raster feature grid into contiguous fixed-size blocks."""
        batch, tokens, channels = feat.shape
        if tokens != height * width:
            raise ValueError(
                f"feature has {tokens} tokens, expected {height}x{width}={height * width}"
            )
        if height % block_size or width % block_size:
            raise ValueError(
                f"1/8 feature shape {(height, width)} must be divisible by block_size={block_size}"
            )
        block_h, block_w = height // block_size, width // block_size
        return (
            feat.view(batch, block_h, block_size, block_w, block_size, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(batch, block_h * block_w, block_size * block_size, channels)
        )

    def _block_token_indices(self, height, width, block_size, device):
        key = ("source", height, width, block_size, str(device))
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached
        block_h, block_w = height // block_size, width // block_size
        indices = (
            torch.arange(height * width, device=device)
            .view(height, width)
            .view(block_h, block_size, block_w, block_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(block_h * block_w, block_size * block_size)
        )
        self._layout_cache[key] = indices
        return indices

    def _routed_target_indices(self, route_idx, height, width):
        """Expand routed block ids to a fixed target patch, optionally with halo."""
        block_size = self.block_size
        halo = self.block_halo
        block_w = width // block_size
        patch_size = block_size + 2 * halo

        key = (
            "target",
            height,
            width,
            block_size,
            halo,
            str(route_idx.device),
        )
        cached = self._layout_cache.get(key)
        if cached is None:
            block_ids = torch.arange(
                (height // block_size) * block_w,
                device=route_idx.device,
            )
            block_y = (
                torch.div(block_ids, block_w, rounding_mode="floor")
                * block_size
            )
            block_x = (block_ids % block_w) * block_size
            offsets = torch.arange(
                -halo, block_size + halo, device=route_idx.device
            )
            target_y = block_y[:, None, None] + offsets.view(
                1, patch_size, 1
            )
            target_x = block_x[:, None, None] + offsets.view(
                1, 1, patch_size
            )
            valid = (
                (target_y >= 0)
                & (target_y < height)
                & (target_x >= 0)
                & (target_x < width)
            )
            target_y = target_y.clamp(0, height - 1)
            target_x = target_x.clamp(0, width - 1)
            patch_indices = (target_y * width + target_x).view(
                block_ids.numel(), patch_size * patch_size
            )
            patch_valid = valid.view(
                block_ids.numel(), patch_size * patch_size
            )
            cached = patch_indices, patch_valid
            self._layout_cache[key] = cached
        patch_indices, patch_valid = cached
        return patch_indices[route_idx], patch_valid[route_idx]

    @staticmethod
    def _sort_unique_target_indices(target_indices, target_valid):
        """Sort and mask duplicate target ids while retaining a fixed tensor shape."""
        # Sort valid copies before invalid clamped copies of the same target id.
        # Otherwise a boundary halo can keep an invalid duplicate and suppress
        # the later valid candidate.
        sort_key = target_indices * 2 + (~target_valid).to(target_indices.dtype)
        _, order = torch.sort(sort_key, dim=-1)
        target_indices = torch.gather(target_indices, -1, order)
        target_valid = torch.gather(target_valid, -1, order)
        unique = torch.ones_like(target_valid)
        unique[..., 1:] = target_indices[..., 1:] != target_indices[..., :-1]
        return target_indices, target_valid & unique

    def _compute_route_scores(
        self,
        feat_r0,
        feat_r1,
        height0,
        width0,
        mask_c0=None,
        mask_c1=None,
    ):
        if feat_r0 is None or feat_r1 is None:
            raise ValueError("router supervision requires 1/32 route features")

        batch = feat_r0.shape[0]
        route_h0, route_w0 = feat_r0.shape[-2:]
        route_h1, route_w1 = feat_r1.shape[-2:]
        expected_route_shape = (
            height0 // self.block_size,
            width0 // self.block_size,
        )
        if (route_h0, route_w0) != expected_route_shape or (
            route_h1,
            route_w1,
        ) != expected_route_shape:
            raise ValueError(
                "route feature shape must align with block grid: "
                f"got {(route_h0, route_w0)} and {(route_h1, route_w1)}, "
                f"expected {expected_route_shape}"
            )

        if self.router is None:
            route0 = F.normalize(
                feat_r0.flatten(2).transpose(1, 2), p=2, dim=-1
            )
            route1 = F.normalize(
                feat_r1.flatten(2).transpose(1, 2), p=2, dim=-1
            )
        else:
            route0, route1 = self.router(feat_r0, feat_r1)

        route_scores = torch.matmul(route0, route1.transpose(1, 2))
        if self.router is not None:
            route_scores = route_scores / self.router_temperature

        if mask_c0 is not None:
            route_mask_value = torch.finfo(route_scores.dtype).min
            route_mask0 = F.max_pool2d(
                mask_c0.view(batch, 1, height0, width0).float(),
                kernel_size=self.block_size,
                stride=self.block_size,
            ).flatten(1).bool()
            route_mask1 = F.max_pool2d(
                mask_c1.view(batch, 1, height0, width0).float(),
                kernel_size=self.block_size,
                stride=self.block_size,
            ).flatten(1).bool()
            route_scores = route_scores.masked_fill(
                ~route_mask1[:, None, :], route_mask_value
            )
            route_scores = route_scores.masked_fill(
                ~route_mask0[:, :, None], route_mask_value
            )

        return route_scores

    def _build_router_positive_mask(self, route_scores, data):
        """Map sparse 1/8 GT correspondences to valid block-pair labels."""
        batch, num_blocks0, num_blocks1 = route_scores.shape
        height0, width0 = data["hw0_c"]
        height1, width1 = data["hw1_c"]
        route_width0 = width0 // self.block_size
        route_width1 = width1 // self.block_size

        b_ids = data["spv_b_ids"]
        i_ids = data["spv_i_ids"]
        j_ids = data["spv_j_ids"]
        gt_valid = data.get(
            "spv_gt_mask", torch.ones_like(b_ids, dtype=torch.bool)
        ).bool()
        gt_valid = gt_valid & (b_ids >= 0) & (b_ids < batch)

        source_y = torch.div(i_ids, width0, rounding_mode="floor")
        source_x = i_ids % width0
        target_y = torch.div(j_ids, width1, rounding_mode="floor")
        target_x = j_ids % width1
        source_blocks = (
            torch.div(source_y, self.block_size, rounding_mode="floor")
            * route_width0
            + torch.div(source_x, self.block_size, rounding_mode="floor")
        )
        target_blocks = (
            torch.div(target_y, self.block_size, rounding_mode="floor")
            * route_width1
            + torch.div(target_x, self.block_size, rounding_mode="floor")
        )
        gt_valid = (
            gt_valid
            & (source_blocks >= 0)
            & (source_blocks < num_blocks0)
            & (target_blocks >= 0)
            & (target_blocks < num_blocks1)
        )

        # Sparse supervision can retain padded correspondences whose loss
        # weight is zero in the dense coarse objective. They must not become
        # router positives after the corresponding block pair has been masked
        # to float32 minimum, otherwise the multi-positive CE is ~3e38 or inf.
        # Clamp indices only for this safe lookup; the range mask above remains
        # authoritative for whether the item is retained.
        safe_b = b_ids.clamp(0, batch - 1)
        safe_source = source_blocks.clamp(0, num_blocks0 - 1)
        safe_target = target_blocks.clamp(0, num_blocks1 - 1)
        route_mask_value = torch.finfo(route_scores.dtype).min
        gt_valid = gt_valid & (
            route_scores[safe_b, safe_source, safe_target] > route_mask_value
        )

        positive = torch.zeros_like(route_scores, dtype=torch.bool)
        if gt_valid.any():
            positive[
                b_ids[gt_valid], source_blocks[gt_valid], target_blocks[gt_valid]
            ] = True

        return positive, b_ids, source_blocks, target_blocks, gt_valid

    def _compute_router_objective(self, route_scores, data):
        """Bidirectional multi-positive block loss and match-weighted recall."""
        batch, num_blocks0, num_blocks1 = route_scores.shape
        (
            positive,
            b_ids,
            source_blocks,
            target_blocks,
            gt_valid,
        ) = self._build_router_positive_mask(route_scores, data)

        def multi_positive_loss(scores, positive_mask):
            valid_sources = positive_mask.any(dim=-1)
            if not valid_sources.any():
                return scores.reshape(-1)[0] * 0.0
            # Select supervised source rows before logsumexp. Computing the
            # numerator on an unsupervised row gives logsumexp(-inf), whose
            # discarded forward value still has an undefined 0/0 backward.
            scores = scores[valid_sources]
            positive_mask = positive_mask[valid_sources]
            log_denominator = torch.logsumexp(scores, dim=-1)
            log_numerator = torch.logsumexp(
                scores.masked_fill(~positive_mask, -torch.inf), dim=-1
            )
            return (log_denominator - log_numerator).mean()

        loss01 = multi_positive_loss(route_scores, positive)
        loss10 = multi_positive_loss(
            route_scores.transpose(1, 2), positive.transpose(1, 2)
        )
        route_loss = 0.5 * (loss01 + loss10)

        route_topk = min(self.route_topk, num_blocks1, num_blocks0)
        indices01 = torch.topk(route_scores, route_topk, dim=-1).indices
        indices10 = torch.topk(
            route_scores.transpose(1, 2), route_topk, dim=-1
        ).indices
        if gt_valid.any():
            valid_b = b_ids[gt_valid]
            valid_source = source_blocks[gt_valid]
            valid_target = target_blocks[gt_valid]
            recall01 = (
                indices01[valid_b, valid_source] == valid_target[:, None]
            ).any(dim=-1).float().mean()
            recall10 = (
                indices10[valid_b, valid_target] == valid_source[:, None]
            ).any(dim=-1).float().mean()
        else:
            recall01 = route_scores.new_zeros(())
            recall10 = route_scores.new_zeros(())

        return route_loss, recall01, recall10

    def _halo_covering_blocks(self, point_ids, height, width):
        """Return route blocks whose halo-expanded patch contains each point."""
        block_size = self.block_size
        halo = self.block_halo
        route_height = height // block_size
        route_width = width // block_size
        radius = math.ceil(halo / block_size)
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=point_ids.device,
            dtype=point_ids.dtype,
        )
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_y = offset_y.flatten()
        offset_x = offset_x.flatten()

        point_y = torch.div(point_ids, width, rounding_mode="floor")
        point_x = point_ids % width
        base_y = torch.div(point_y, block_size, rounding_mode="floor")
        base_x = torch.div(point_x, block_size, rounding_mode="floor")
        candidate_y = base_y[:, None] + offset_y[None]
        candidate_x = base_x[:, None] + offset_x[None]
        valid = (
            (candidate_y >= 0)
            & (candidate_y < route_height)
            & (candidate_x >= 0)
            & (candidate_x < route_width)
            & (point_y[:, None] >= candidate_y * block_size - halo)
            & (
                point_y[:, None]
                < (candidate_y + 1) * block_size + halo
            )
            & (point_x[:, None] >= candidate_x * block_size - halo)
            & (
                point_x[:, None]
                < (candidate_x + 1) * block_size + halo
            )
        )
        candidate_y = candidate_y.clamp(0, route_height - 1)
        candidate_x = candidate_x.clamp(0, route_width - 1)
        return candidate_y * route_width + candidate_x, valid

    @staticmethod
    def _multi_positive_router_loss(scores, positive_mask):
        valid_sources = positive_mask.any(dim=-1)
        if not valid_sources.any():
            return scores.reshape(-1)[0] * 0.0
        selected_scores = scores[valid_sources].float()
        selected_positive = positive_mask[valid_sources]
        log_denominator = torch.logsumexp(selected_scores, dim=-1)
        log_numerator = torch.logsumexp(
            selected_scores.masked_fill(~selected_positive, -torch.inf),
            dim=-1,
        )
        return (log_denominator - log_numerator).mean()

    def _directional_objective_by_size(
        self,
        scores,
        b_ids,
        source_blocks,
        exact_target_blocks,
        target_points,
        target_height,
        target_width,
    ):
        """Halo-aware CE, Top-K margin loss, and recalls for one direction."""
        zero = scores.reshape(-1)[0] * 0.0
        metrics = {}
        if b_ids.numel() == 0:
            for route_size in self.train_route_sizes:
                metrics[route_size] = (zero.detach(), zero.detach())
            return zero, zero, metrics

        candidates, candidate_valid = self._halo_covering_blocks(
            target_points, target_height, target_width
        )
        candidate_scores = scores[b_ids[:, None], source_blocks[:, None], candidates]
        mask_value = torch.finfo(scores.dtype).min
        candidate_valid = candidate_valid & (candidate_scores > mask_value)

        halo_positive = torch.zeros_like(scores, dtype=torch.bool)
        if candidate_valid.any():
            expanded_b = b_ids[:, None].expand_as(candidates)
            expanded_source = source_blocks[:, None].expand_as(candidates)
            halo_positive[
                expanded_b[candidate_valid],
                expanded_source[candidate_valid],
                candidates[candidate_valid],
            ] = True
        coverage_loss = self._multi_positive_router_loss(
            scores, halo_positive
        )

        max_route_size = min(max(self.train_route_sizes), scores.shape[-1])
        top_values, top_indices = torch.topk(
            scores, max_route_size, dim=-1
        )
        match_top_values = top_values[b_ids, source_blocks].float()
        match_top_indices = top_indices[b_ids, source_blocks]
        best_covering_score = candidate_scores.float().masked_fill(
            ~candidate_valid, -torch.inf
        ).amax(dim=-1)
        valid_matches = torch.isfinite(best_covering_score)

        rank_losses = []
        for route_size in self.train_route_sizes:
            effective_route_size = min(route_size, max_route_size)
            selected = match_top_indices[:, :effective_route_size]
            exact_recall = (
                selected == exact_target_blocks[:, None]
            ).any(dim=-1).float().mean()
            halo_matches = (
                (selected[:, :, None] == candidates[:, None, :])
                & candidate_valid[:, None, :]
            )
            halo_recall = halo_matches.flatten(1).any(dim=1).float().mean()
            metrics[route_size] = (exact_recall.detach(), halo_recall.detach())

            if valid_matches.any():
                boundary = match_top_values[valid_matches, effective_route_size - 1]
                positive = best_covering_score[valid_matches]
                rank_losses.append(
                    F.softplus(
                        boundary - positive + self.router_rank_margin
                    ).mean()
                )

        rank_loss = (
            torch.stack(rank_losses).mean() if rank_losses else zero
        )
        return coverage_loss, rank_loss, metrics

    def _compute_router_objective_by_size(self, route_scores, data):
        """Compute geometry-supervised objectives for each training route size."""
        (
            _,
            b_ids,
            source_blocks,
            target_blocks,
            gt_valid,
        ) = self._build_router_positive_mask(route_scores, data)
        b_ids = b_ids[gt_valid]
        source_blocks = source_blocks[gt_valid]
        target_blocks = target_blocks[gt_valid]
        source_points = data["spv_i_ids"][gt_valid]
        target_points = data["spv_j_ids"][gt_valid]
        height0, width0 = data["hw0_c"]
        height1, width1 = data["hw1_c"]

        loss01, rank01, metrics01 = self._directional_objective_by_size(
            route_scores,
            b_ids,
            source_blocks,
            target_blocks,
            target_points,
            height1,
            width1,
        )
        loss10, rank10, metrics10 = self._directional_objective_by_size(
            route_scores.transpose(1, 2),
            b_ids,
            target_blocks,
            source_blocks,
            source_points,
            height0,
            width0,
        )
        metrics = {
            route_size: {
                "exact01": metrics01[route_size][0],
                "halo01": metrics01[route_size][1],
                "exact10": metrics10[route_size][0],
                "halo10": metrics10[route_size][1],
            }
            for route_size in self.train_route_sizes
        }
        return 0.5 * (loss01 + loss10), 0.5 * (rank01 + rank10), metrics

    @torch.no_grad()
    def _inject_gt_routes(self, route_idx, route_scores, data):
        """Replace a shrinking prefix of Top-R slots with valid GT blocks."""
        if not self.training or self.gt_inject_epochs <= 0:
            return route_idx, 0

        epoch_progress = float(data.get("train_epoch_progress", 0.0))
        remaining = max(
            0.0, 1.0 - epoch_progress / self.gt_inject_epochs
        )
        if remaining <= 0:
            return route_idx, 0

        route_size = route_idx.shape[-1]
        injected_slots = min(
            route_size,
            max(1, math.ceil(route_size * remaining)),
        )
        positive, _, _, _, _ = self._build_router_positive_mask(
            route_scores, data
        )
        gt_values, gt_targets = torch.topk(
            positive.to(torch.uint8), injected_slots, dim=-1
        )
        injected_prefix = torch.where(
            gt_values.bool(),
            gt_targets,
            route_idx[..., :injected_slots],
        )
        route_idx = torch.cat(
            [injected_prefix, route_idx[..., injected_slots:]], dim=-1
        )
        return route_idx, injected_slots

    def forward_router_supervision(
        self,
        feat_r0,
        feat_r1,
        data,
        mask_c0=None,
        mask_c1=None,
    ):
        route_scores = self._compute_route_scores(
            feat_r0,
            feat_r1,
            data["hw0_c"][0],
            data["hw0_c"][1],
            mask_c0=mask_c0,
            mask_c1=mask_c1,
        )
        route_loss, recall01, recall10 = self._compute_router_objective(
            route_scores, data
        )
        router_outputs = {
            "loss_router": route_loss,
            "route_recall01": recall01.detach(),
            "route_recall10": recall10.detach(),
        }
        if self.train_route_sizes:
            coverage_loss, rank_loss, metrics_by_size = (
                self._compute_router_objective_by_size(
                    route_scores, data
                )
            )
            router_outputs.update(
                {
                    "loss_router_coverage": coverage_loss,
                    "loss_router_rank": rank_loss,
                }
            )
            for route_size, metrics in metrics_by_size.items():
                router_outputs.update(
                    {
                        f"route_recall01_r{route_size}": metrics["exact01"],
                        f"route_recall10_r{route_size}": metrics["exact10"],
                        f"route_halo_recall01_r{route_size}": metrics["halo01"],
                        f"route_halo_recall10_r{route_size}": metrics["halo10"],
                    }
                )
        data.update(router_outputs)
        return route_scores

    def _compute_routes(
        self,
        feat_r0,
        feat_r1,
        height0,
        width0,
        mask_c0=None,
        mask_c1=None,
        route_scores=None,
        route_topk=None,
    ):
        if route_scores is None:
            route_scores = self._compute_route_scores(
                feat_r0,
                feat_r1,
                height0,
                width0,
                mask_c0=mask_c0,
                mask_c1=mask_c1,
            )
        if route_topk is None:
            route_topk = self.route_topk
        route_topk = min(int(route_topk), route_scores.shape[-1])
        route_values, route_idx = torch.topk(route_scores, route_topk, dim=-1)
        return route_values, route_idx, route_scores

    def _select_packed_confidence(
        self, conf_matrix, target_indices, source_indices, data
    ):
        """Select the fixed Top-K from a packed candidate confidence tensor."""
        row_max_val, row_max_pos = conf_matrix.max(dim=-1)
        self._select_packed_row_max(
            row_max_val, row_max_pos, target_indices, source_indices, data
        )

    def _select_packed_row_max(
        self,
        row_max_val,
        row_max_pos,
        target_indices,
        source_indices,
        data,
    ):
        """Finish packed Top-K selection from per-source row maxima."""
        batch, _, source_per_block = row_max_val.shape
        selected_target = torch.gather(
            target_indices[:, :, None, :].expand(
                -1, -1, source_per_block, -1
            ),
            3,
            row_max_pos[..., None],
        ).squeeze(-1)

        k = self.topk
        if k == -1 or k > row_max_val[0].numel():
            k = row_max_val[0].numel()
        topk_val, topk_pos = torch.topk(row_max_val.flatten(1), k, dim=1)
        source_lookup = source_indices.flatten()[None].expand(batch, -1)
        i_ids = torch.gather(source_lookup, 1, topk_pos).flatten()
        j_ids = torch.gather(selected_target.flatten(1), 1, topk_pos).flatten()
        b_ids = (
            torch.arange(batch, device=row_max_val.device)
            .unsqueeze(1)
            .expand(-1, k)
            .flatten()
        )
        self._update_matches(data, b_ids, i_ids, j_ids, topk_val.flatten())

    def _compute_packed_similarity(
        self,
        source_blocks,
        target_features,
        target_indices,
    ):
        """Chunk target gathers to avoid materializing all routed features."""
        batch, blocks, source_per_block, channels = source_blocks.shape
        candidates = target_indices.shape[-1]
        chunk_size = max(1, min(self.block_chunk_size, blocks))
        similarity = torch.empty(
            batch,
            blocks,
            source_per_block,
            candidates,
            dtype=source_blocks.dtype,
            device=source_blocks.device,
        )
        scale = math.sqrt(channels)
        prescale = not torch.is_grad_enabled()
        # Routed targets are often gathered repeatedly by neighboring source
        # blocks. Scale each descriptor once before those duplicate gathers in
        # inference, while retaining the original autograd graph in training.
        source_input = source_blocks / scale if prescale else source_blocks
        target_input = target_features / scale if prescale else target_features
        for start in range(0, blocks, chunk_size):
            end = min(start + chunk_size, blocks)
            chunk_indices = target_indices[:, start:end]
            packed_target = torch.gather(
                target_input,
                1,
                chunk_indices.flatten(1)[..., None].expand(-1, -1, channels),
            ).view(batch, end - start, candidates, channels)
            if prescale:
                chunk_similarity = torch.matmul(
                    source_input[:, start:end],
                    packed_target.transpose(-1, -2),
                )
            else:
                chunk_similarity = torch.matmul(
                    source_input[:, start:end] / scale,
                    (packed_target / scale).transpose(-1, -2),
                )
            similarity[:, start:end] = chunk_similarity / self.temperature
        return similarity

    def _apply_route_score_prior(
        self,
        similarity,
        route_scores,
        target_indices,
        target_valid,
        height1,
        width1,
        weight_scale=1.0,
    ):
        """Inject one already-computed block logit into each packed edge."""
        if not self.router_score_prior:
            return similarity
        if self.router is None:
            raise RuntimeError("Router score prior requires a trainable router")

        route_width1 = width1 // self.block_size
        target_y = torch.div(
            target_indices, width1, rounding_mode="floor"
        )
        target_x = target_indices % width1
        target_blocks = (
            torch.div(target_y, self.block_size, rounding_mode="floor")
            * route_width1
            + torch.div(target_x, self.block_size, rounding_mode="floor")
        )
        packed_prior = torch.gather(route_scores, 2, target_blocks)

        # A fully padded source block has finfo.min router logits. It is later
        # masked out of sparse dual-softmax, but must be neutralized before the
        # learnable multiplication to keep its forward/backward finite.
        valid_prior = target_valid & (
            packed_prior > torch.finfo(packed_prior.dtype).min / 2
        )
        packed_prior = torch.where(
            valid_prior, packed_prior, torch.zeros_like(packed_prior)
        )
        # tanh bounds the prior weight and has unit slope at zero.
        weight = self.router_score_prior_max * torch.tanh(
            self.router.score_prior_weight
        )
        weight = weight.to(similarity.dtype) * weight_scale
        return similarity.add(
            packed_prior.to(similarity.dtype).unsqueeze(2) * weight
        )

    def _collect_routed_gt_confidence(
        self,
        confidence,
        target_indices,
        target_valid,
        data,
    ):
        """Collect sparse confidence values for GT matches covered by routing."""
        valid = data.get(
            "spv_gt_mask",
            torch.ones_like(data["spv_b_ids"], dtype=torch.bool),
        ).bool()
        b_ids = data["spv_b_ids"][valid]
        i_ids = data["spv_i_ids"][valid]
        j_ids = data["spv_j_ids"][valid]
        if b_ids.numel() == 0:
            data["routed_gt_confidence"] = confidence.reshape(-1)[:0]
            data["routed_gt_recall"] = confidence.new_zeros(())
            return

        width0 = data["hw0_c"][1]
        source_y = torch.div(i_ids, width0, rounding_mode="floor")
        source_x = i_ids % width0
        block_width = width0 // self.block_size
        source_blocks = (
            torch.div(source_y, self.block_size, rounding_mode="floor")
            * block_width
            + torch.div(source_x, self.block_size, rounding_mode="floor")
        )
        source_local = (source_y % self.block_size) * self.block_size
        source_local = source_local + source_x % self.block_size

        gt_candidates = target_indices[b_ids, source_blocks]
        gt_candidate_valid = target_valid[b_ids, source_blocks]
        matches = (gt_candidates == j_ids[:, None]) & gt_candidate_valid
        covered = matches.any(dim=-1)
        candidate_position = matches.to(torch.int64).argmax(dim=-1)
        covered_confidence = confidence[
            b_ids,
            source_blocks,
            source_local,
            candidate_position,
        ][covered]
        data["routed_gt_confidence"] = covered_confidence
        data["routed_gt_recall"] = covered.float().mean().detach()

    def _forward_block_routed(
        self,
        feat_c0,
        feat_c1,
        feat_r0,
        feat_r1,
        data,
        mask_c0=None,
        mask_c1=None,
        route_scores=None,
    ):
        """Select a fixed number of target blocks at 1/32 and match their tokens at 1/8."""
        batch, tokens0, channels = feat_c0.shape
        tokens1 = feat_c1.shape[1]
        height0, width0 = data["hw0_c"]
        height1, width1 = data["hw1_c"]
        block_size = self.block_size
        route_h0, route_w0 = feat_r0.shape[-2:]

        if (height0, width0) != (height1, width1):
            raise ValueError("block_routed mode currently requires equal 1/8 feature shapes")
        if tokens0 != tokens1:
            raise ValueError("block_routed mode currently requires equal token counts")

        active_route_size = (
            int(data["train_route_size"])
            if self.training and "train_route_size" in data
            else self.route_topk
        )
        route_values, route_idx, route_scores = self._compute_routes(
            feat_r0,
            feat_r1,
            height0,
            width0,
            mask_c0=mask_c0,
            mask_c1=mask_c1,
            route_scores=route_scores,
            route_topk=active_route_size,
        )
        data["route_size_used"] = route_scores.new_tensor(
            route_idx.shape[-1]
        )
        route_idx, injected_slots = self._inject_gt_routes(
            route_idx, route_scores, data
        )
        if injected_slots:
            route_values = torch.gather(route_scores, -1, route_idx)
        data["gt_route_injected_slots"] = route_scores.new_tensor(
            injected_slots
        )

        source_blocks = self._to_blocks(
            feat_c0, height0, width0, block_size
        )
        target_indices, target_valid = self._routed_target_indices(
            route_idx, height1, width1
        )
        target_indices, target_valid = self._sort_unique_target_indices(
            target_indices.flatten(2), target_valid.flatten(2)
        )
        sim_matrix = self._compute_packed_similarity(
            source_blocks, feat_c1, target_indices
        )
        sim_matrix = self._apply_route_score_prior(
            sim_matrix,
            route_scores,
            target_indices,
            target_valid,
            height1,
            width1,
            # Injected ground-truth blocks can have low router scores. A zero scale
            # disables the prior during injection and keeps its parameter in the DDP graph.
            weight_scale=(
                0.0 if self.training and injected_slots > 0 else 1.0
            ),
        )
        if self.router_score_prior:
            prior_active = not (self.training and injected_slots > 0)
            data["route_score_prior_weight"] = (
                self.router_score_prior_max
                * torch.tanh(self.router.score_prior_weight.detach())
                * float(prior_active)
            )
            data["route_score_prior_active"] = route_scores.new_tensor(
                float(prior_active)
            )

        source_indices = self._block_token_indices(
            height0, width0, block_size, feat_c0.device
        )
        if mask_c0 is not None:
            source_valid = torch.gather(
                mask_c0,
                1,
                source_indices.flatten()[None].expand(batch, -1),
            ).view(batch, route_h0 * route_w0, block_size * block_size)
            target_valid = target_valid & torch.gather(
                mask_c1,
                1,
                target_indices.flatten(1),
            ).view_as(target_valid).bool()
        else:
            source_valid = torch.ones(
                batch,
                route_h0 * route_w0,
                block_size * block_size,
                dtype=torch.bool,
                device=feat_c0.device,
            )

        candidate_valid = source_valid[..., None] & target_valid[:, :, None, :]
        fused_selection = False
        if self.mode == "block_sparse":
            sparse_scores = sim_matrix.float().masked_fill(
                ~candidate_valid, -torch.inf
            )
            use_l1_dual_softmax = not self.training and self.ds_opt
            if use_l1_dual_softmax:
                # Exponentiate once and reuse
                # the same positive tensor for both L1 normalizations. The
                # target normalization below is still global over every routed
                # edge incident on a target token, rather than local to a
                # packed block. Invalid candidates are -inf and become exact 0.
                sparse_exp = torch.exp(sparse_scores)
                row_probability = F.normalize(
                    sparse_exp, p=1, dim=-1
                )
                column_exp = sparse_exp
            else:
                # Training (and DS_OPT=False evaluation) keeps independently
                # stabilized softmax factors. A single global shift can
                # underflow an otherwise valid low-scoring row/column and make
                # its backward denominator overflow. Each non-empty row below
                # contains exp(0)=1, while empty masked rows remain exact 0.
                row_max = sparse_scores.amax(dim=-1, keepdim=True).detach()
                row_max = torch.where(
                    torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
                )
                row_exp = torch.exp(sparse_scores - row_max)
                row_sum = row_exp.sum(dim=-1, keepdim=True)
                safe_row_sum = torch.where(
                    row_sum > 0, row_sum, torch.ones_like(row_sum)
                )
                row_probability = row_exp / safe_row_sum

            deterministic = torch.are_deterministic_algorithms_enabled()
            connection_reduce = deterministic or (
                self.fused_sparse_selection and not self.training
            )
            if connection_reduce:
                connection_targets = target_indices.flatten(1)
                if deterministic:
                    connection_groups = self._deterministic_index_groups(
                        connection_targets, tokens1
                    )
                if not use_l1_dual_softmax:
                    # First reduce source tokens inside each routed block. This
                    # shrinks the stable segmented column-max reduction by
                    # block_size^2 while retaining the same target-token max.
                    connection_max = sparse_scores.amax(dim=2).flatten(1)
                    if deterministic:
                        column_max = self._deterministic_index_reduce(
                            connection_max,
                            connection_targets,
                            tokens1,
                            "max",
                            groups=connection_groups,
                        )
                    else:
                        column_max = torch.full(
                            (batch, tokens1),
                            -torch.inf,
                            dtype=sparse_scores.dtype,
                            device=sparse_scores.device,
                        )
                        column_max.scatter_reduce_(
                            1,
                            connection_targets,
                            connection_max,
                            reduce="amax",
                            include_self=True,
                        )
                    candidate_column_max = torch.gather(
                        column_max, 1, connection_targets
                    ).view(
                        batch,
                        target_indices.shape[1],
                        1,
                        target_indices.shape[-1],
                    )
            else:
                expanded_targets = target_indices[:, :, None, :].expand(
                    -1, -1, block_size * block_size, -1
                )
                flat_targets = expanded_targets.flatten(1)
                if not use_l1_dual_softmax:
                    flat_scores = sparse_scores.flatten(1)
                    column_max = torch.full(
                        (batch, tokens1),
                        -torch.inf,
                        dtype=sparse_scores.dtype,
                        device=sparse_scores.device,
                    )
                    column_max.scatter_reduce_(
                        1,
                        flat_targets,
                        flat_scores,
                        reduce="amax",
                        include_self=True,
                    )
                    candidate_column_max = torch.gather(
                        column_max, 1, flat_targets
                    ).view_as(sparse_scores)
            if not use_l1_dual_softmax:
                column_max = column_max.detach()
                candidate_column_max = candidate_column_max.detach()
                candidate_column_max = torch.where(
                    torch.isfinite(candidate_column_max),
                    candidate_column_max,
                    torch.zeros_like(candidate_column_max),
                )
                column_exp = torch.exp(
                    sparse_scores - candidate_column_max
                ) * candidate_valid
            if connection_reduce:
                connection_sum = column_exp.sum(dim=2).flatten(1)
                if deterministic:
                    column_sum = self._deterministic_index_reduce(
                        connection_sum,
                        connection_targets,
                        tokens1,
                        "sum",
                        groups=connection_groups,
                    )
                else:
                    column_sum = torch.zeros(
                        batch,
                        tokens1,
                        dtype=sparse_scores.dtype,
                        device=sparse_scores.device,
                    )
                    column_sum.scatter_add_(
                        1, connection_targets, connection_sum
                    )
                candidate_column_sum = torch.gather(
                    column_sum, 1, connection_targets
                ).view(
                    batch,
                    target_indices.shape[1],
                    1,
                    target_indices.shape[-1],
                )
            else:
                flat_column_exp = column_exp.flatten(1)
                column_sum = torch.zeros(
                    batch,
                    tokens1,
                    dtype=sparse_scores.dtype,
                    device=sparse_scores.device,
                )
                column_sum.scatter_add_(
                    1, flat_targets, flat_column_exp
                )
                candidate_column_sum = torch.gather(
                    column_sum, 1, flat_targets
                ).view_as(sparse_scores)
            safe_column_sum = torch.where(
                candidate_column_sum > 0,
                candidate_column_sum,
                torch.ones_like(candidate_column_sum),
            )
            column_probability = column_exp / safe_column_sum
            fused_selection = (
                self.fused_sparse_selection
                and not self.training
                and data.get("spv_b_ids") is None
            )
            if fused_selection:
                row_max_val, row_max_pos = fused_confidence_rowmax(
                    row_probability,
                    column_probability,
                    candidate_valid,
                )
                conf_matrix = None
            else:
                conf_matrix = (
                    row_probability * column_probability * candidate_valid
                )
        else:
            sim_matrix.masked_fill_(
                ~candidate_valid, torch.finfo(sim_matrix.dtype).min
            )
            conf_matrix = F.softmax(sim_matrix, dim=-2) * F.softmax(
                sim_matrix, dim=-1
            )
            conf_matrix = conf_matrix * candidate_valid
        if data.get("spv_b_ids", None) is not None:
            self._collect_routed_gt_confidence(
                conf_matrix, target_indices, target_valid, data
            )

        if fused_selection:
            self._select_packed_row_max(
                row_max_val,
                row_max_pos,
                target_indices,
                source_indices,
                data,
            )
        else:
            self._select_packed_confidence(
                conf_matrix, target_indices, source_indices, data
            )
        data.update(
            {
                "block_route_indices": route_idx,
                "block_route_scores": route_values.detach(),
                "coarse_mode": self.mode,
            }
        )
        if data.get("spv_w_pt0_i", None) is not None:
            pad_num = self.pad_num if self.training else 0
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_fine(data, pad_num, self.window_size)
        return None

    @staticmethod
    def _update_matches(data, b_ids, i_ids, j_ids, mconf):
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
        mkpts0_c = (
            torch.stack(
                [
                    i_ids % data["hw0_c"][1],
                    torch.div(i_ids, data["hw0_c"][1], rounding_mode="floor"),
                ],
                dim=1,
            )
            * scale0
        )
        mkpts1_c = (
            torch.stack(
                [
                    j_ids % data["hw1_c"][1],
                    torch.div(j_ids, data["hw1_c"][1], rounding_mode="floor"),
                ],
                dim=1,
            )
            * scale1
        )
        data.update(
            {
                "mconf": mconf,
                "mkpts0_c": mkpts0_c,
                "mkpts1_c": mkpts1_c,
                "b_ids": b_ids,
                "i_ids": i_ids,
                "j_ids": j_ids,
            }
        )

    @torch.no_grad()
    def coarse_matching_selection(self, data):
        """Store the top-scoring row maxima as coarse matches for each image pair."""
        conf_matrix = data["conf_matrix"]

        k = self.topk
        row_max_val, row_max_idx = torch.max(conf_matrix, dim=2)

        # prevent out of range
        if k == -1 or k > row_max_val.shape[-1]:
            k = row_max_val.shape[-1]

        topk_val, topk_idx = torch.topk(row_max_val, k)
        b_ids = (
            torch.arange(conf_matrix.shape[0], device=conf_matrix.device)
            .unsqueeze(1)
            .repeat(1, k)
            .flatten()
        )
        i_ids = topk_idx.flatten()
        j_ids = row_max_idx[b_ids, i_ids].flatten()
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        self._update_matches(data, b_ids, i_ids, j_ids, mconf)

        # Fine matching supervision during the training process
        if data.get("spv_w_pt0_i", None) is not None:
            pad_num = self.pad_num if self.training else 0
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_fine(data, pad_num, self.window_size)
