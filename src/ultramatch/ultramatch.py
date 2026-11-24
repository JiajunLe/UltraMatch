import contextlib

from ..utils.checkpoint import rename_state_dict_prefix
from ..utils.misc import detect_NaN
from .head.fine_matching import TinyFineMatching
from .head.coarse_matching import CoarseMatching
from .neck.neck import CIM
from .backbone.resnet import ResNet18
from .utils.reparameter import reparameterize_conv_bn
from einops.einops import rearrange
import torch.nn as nn
import torch
torch.set_float32_matmul_precision("highest")


class UltraMatch(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.local_resolution = self.config["local_resolution"]
        self.deploy = self.config["deploy"]
        self.reparameterize = self.config.get("reparameterize", False)
        self._is_reparameterized = False
        self.feature_amp = self.config.get("feature_amp", "off")
        if self.feature_amp not in {"off", "bf16", "fp16"}:
            raise ValueError(
                "FEATURE_AMP must be one of: off, bf16, fp16"
            )
        self.channels_last = self.config.get("channels_last", False)
        self._channels_last_enabled = False
        self._attention_fused = False
        self.topk = config["coarse"]["topk"]
        self.prefilter_fine = config["coarse"].get("prefilter_fine", False)
        self.prefilter_strategy = config["coarse"].get(
            "prefilter_strategy", "mask"
        )
        self.prefilter_buckets = tuple(
            sorted(config["coarse"].get("prefilter_buckets", [2048, 4096]))
        )
        self.prefilter_fixed_k = config["coarse"].get(
            "prefilter_fixed_k", 4096
        )
        self.mconf_thr = config["coarse"]["mconf_thr"]
        router_config = config.get("router", {})
        self.router_train_only = router_config.get("train_only", False)
        self.router_freeze_features = router_config.get("freeze_features", True)

        # Modules
        self.backbone = ResNet18(config)
        self.neck = CIM(config)
        self.coarse_matching = CoarseMatching(config)
        self.fine_matching = TinyFineMatching(config)
        self._configure_router_training()

    def _configure_router_training(self):
        if not self.router_train_only:
            return
        if self.coarse_matching.router is None:
            raise ValueError("ROUTER.TRAIN_ONLY requires ROUTER.ENABLED")

        self.requires_grad_(False)
        self.coarse_matching.router.requires_grad_(True)
        if not self.router_freeze_features:
            for module in (
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
                self.backbone.layer5,
            ):
                module.requires_grad_(True)
            self.neck.fc32.requires_grad_(True)
            self.neck.loftr_32.requires_grad_(True)

    def train(self, mode=True):
        if mode and self._is_reparameterized:
            raise RuntimeError(
                "A Conv-BN reparameterized UltraMatch is inference-only; reload the "
                "unfused checkpoint before training"
            )
        if mode and self._attention_fused:
            raise RuntimeError(
                "An attention-reparameterized UltraMatch is inference-only; reload "
                "the unfused checkpoint before training"
            )
        super().train(mode)
        if mode and self.router_train_only and self.router_freeze_features:
            self.backbone.eval()
            self.neck.eval()
            self.coarse_matching.router.train()
        return self

    def reparameterize_for_inference(self):
        """Fold Conv-BN pairs after checkpoint loading for final inference."""
        if self._is_reparameterized:
            return self
        if self.training:
            raise RuntimeError("Call eval() before inference reparameterization")
        # Rep branches must be summed before the generic Conv-BN pass turns
        # their branch BNs into Identity modules.
        self.backbone.switch_to_deploy()
        self.neck.switch_to_deploy()
        reparameterize_conv_bn(self)
        self._is_reparameterized = True
        return self

    def fuse_attention_for_inference(self):
        """Fuse loaded Transformer projections without changing checkpoints."""
        if self._attention_fused:
            return self
        if self.training:
            raise RuntimeError("Call eval() before attention projection fusion")
        self.neck.loftr_32.fuse_projections_for_inference()
        self._attention_fused = True
        return self

    def enable_channels_last_for_inference(self):
        if self._channels_last_enabled:
            return self
        if self.training:
            raise RuntimeError("Call eval() before enabling channels-last")
        self.backbone.to(memory_format=torch.channels_last)
        self.neck.to(memory_format=torch.channels_last)
        self._channels_last_enabled = True
        return self

    def _feature_autocast(self, image):
        enabled = (
            not self.training
            and self.feature_amp != "off"
            and image.device.type == "cuda"
        )
        if not enabled:
            return torch.autocast(device_type=image.device.type, enabled=False)
        if self.feature_amp == "bf16":
            # PyTorch 2.0 checks the current GPU when creating a BF16 context.
            with torch.cuda.device(image.device):
                if torch.cuda.is_bf16_supported():
                    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False)
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _extract_and_interact_features(self, data, mask_c0, mask_c1):
        image0, image1 = data["image0"], data["image1"]
        if self._channels_last_enabled:
            image0 = image0.contiguous(memory_format=torch.channels_last)
            image1 = image1.contiguous(memory_format=torch.channels_last)

        if data["hw0_i"] == data["hw1_i"]:
            # Faster and better BN convergence for equal image shapes.
            feats = self.backbone(torch.cat([image0, image1], dim=0))
            f8, f16, f32, f8_fine = feats
            ms_feats = f8, f16, f32
            feat_f0, feat_f1 = f8_fine.chunk(2)
        else:
            feats0 = self.backbone(image0)
            feats1 = self.backbone(image1)
            f8_0, f16_0, f32_0, feat_f0 = feats0
            f8_1, f16_1, f32_1, feat_f1 = feats1
            ms_feats = f8_0, f16_0, f32_0, f8_1, f16_1, f32_1

        feat_c0, feat_c1, feat_r0, feat_r1 = self.neck(
            ms_feats, mask_c0, mask_c1
        )
        return feat_c0, feat_c1, feat_r0, feat_r1, feat_f0, feat_f1

    def _forward_router_training(self, data, mask_c0=None, mask_c1=None):
        grad_context = (
            torch.no_grad()
            if self.router_freeze_features
            else contextlib.nullcontext()
        )
        with grad_context:
            if data["hw0_i"] == data["hw1_i"]:
                feat32 = self.backbone.forward_route(
                    torch.cat((data["image0"], data["image1"]), dim=0)
                )
                feat_r0, feat_r1 = self.neck.forward_router(
                    feat32, mask_c0=mask_c0, mask_c1=mask_c1
                )
            else:
                feat32_0 = self.backbone.forward_route(data["image0"])
                feat32_1 = self.backbone.forward_route(data["image1"])
                feat_r0, feat_r1 = self.neck.forward_router(
                    feat32_0,
                    feat32_1,
                    mask_c0=mask_c0,
                    mask_c1=mask_c1,
                )

        block_size = self.coarse_matching.block_size
        data.update(
            {
                "hw0_c": (
                    feat_r0.shape[-2] * block_size,
                    feat_r0.shape[-1] * block_size,
                ),
                "hw1_c": (
                    feat_r1.shape[-2] * block_size,
                    feat_r1.shape[-1] * block_size,
                ),
            }
        )
        self.coarse_matching.forward_router_supervision(
            feat_r0,
            feat_r1,
            data,
            mask_c0=mask_c0.flatten(1) if mask_c0 is not None else None,
            mask_c1=mask_c1.flatten(1) if mask_c1 is not None else None,
        )

    @staticmethod
    def _gather_fine_candidate_rows(data, row_indices, original_count):
        """Apply one candidate selection to matches and bidirectional fine GT."""
        for key in (
            "b_ids",
            "i_ids",
            "j_ids",
            "mconf",
            "mkpts0_c",
            "mkpts1_c",
            "offset01_gt",
            "offset10_gt",
        ):
            value = data.get(key)
            if value is None:
                continue
            if value.shape[0] != original_count:
                raise RuntimeError(
                    f"{key} has {value.shape[0]} rows for "
                    f"{original_count} fine candidates"
                )
            data[key] = value[row_indices]

        for key in ("target_uv", "target_uv_weight"):
            value = data.get(key)
            if value is None:
                continue
            if value.shape[0] != 2 * original_count:
                raise RuntimeError(
                    f"{key} has {value.shape[0]} rows for "
                    f"{original_count} bidirectional fine candidates"
                )
            data[key] = torch.cat(
                (
                    value[:original_count][row_indices],
                    value[original_count:][row_indices],
                )
            )

    def _prefilter_fine_candidates(self, data):
        """Prepare mask, compact, bucket, or fixed-K fine candidates."""
        if (
            not self.prefilter_fine
            or self.training
            or self.deploy
        ):
            return
        strategy = self.prefilter_strategy
        if strategy not in {"mask", "dynamic", "bucket", "fixed"}:
            raise ValueError(f"Unknown fine prefilter strategy: {strategy}")

        confidence = data["mconf"]
        keep = confidence > self.mconf_thr
        if strategy == "mask" or data["bs"] != 1:
            data["fine_candidate_mask"] = keep
            return

        original_count = confidence.numel()
        if strategy == "fixed":
            output_count = min(self.prefilter_fixed_k, original_count)
            row_indices = torch.topk(
                confidence, output_count, sorted=True
            ).indices
            self._gather_fine_candidate_rows(
                data, row_indices, original_count
            )
            data["fine_candidate_mask"] = (
                data["mconf"] > self.mconf_thr
            )
            return

        selected = torch.nonzero(keep, as_tuple=False).flatten()
        selected_count = selected.numel()
        if strategy == "dynamic":
            if selected_count == 0:
                data["fine_candidate_mask"] = keep
                return
            self._gather_fine_candidate_rows(
                data, selected, original_count
            )
            data["fine_candidate_mask"] = torch.ones_like(
                data["mconf"], dtype=torch.bool
            )
            return

        if not self.prefilter_buckets:
            raise ValueError("PREFILTER_BUCKETS must not be empty")
        output_count = next(
            (
                bucket
                for bucket in self.prefilter_buckets
                if bucket >= selected_count
            ),
            original_count,
        )
        output_count = max(1, output_count)
        if output_count < selected_count:
            output_count = selected_count
        pad_count = output_count - selected_count
        if selected_count:
            pad_index = selected[:1].expand(pad_count)
        else:
            pad_index = torch.zeros(
                pad_count, dtype=torch.long, device=confidence.device
            )
        row_indices = torch.cat((selected, pad_index))
        self._gather_fine_candidate_rows(data, row_indices, original_count)
        data["fine_candidate_mask"] = torch.arange(
            output_count, device=confidence.device
        ) < selected_count

    def forward(self, data):
        """Match grayscale image pairs and store the predictions in the input batch.

        The batch contains image0 and image1 with shape (N, 1, H, W). Optional
        mask0 and mask1 mark valid cells on each coarse feature grid. Inference
        writes matched points to mkpts0_f and mkpts1_f and confidence to mconf.
        Deployment mode accepts a stacked image pair and returns packed matches.
        """
        if self.deploy:
            image0, image1 = data.split(1, 1)
            data = {"image0": image0, "image1": image1}

        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        mask_c0 = mask_c1 = None
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]

        if self.router_train_only and self.training:
            self._forward_router_training(data, mask_c0, mask_c1)
            return

        # 1-2. Feature extraction and interaction. Selective AMP ends here:
        # routing, sparse Dual-Softmax, candidate selection and Fine remain
        # FP32 even when the expensive Conv2d/SDPA feature path uses BF16/FP16.
        with self._feature_autocast(data["image0"]):
            (
                feat_c0,
                feat_c1,
                feat_r0,
                feat_r1,
                feat_f0,
                feat_f1,
            ) = self._extract_and_interact_features(data, mask_c0, mask_c1)

        if self.feature_amp != "off" and not self.training:
            feat_c0, feat_c1, feat_r0, feat_r1, feat_f0, feat_f1 = (
                feature.float()
                for feature in (
                    feat_c0,
                    feat_c1,
                    feat_r0,
                    feat_r1,
                    feat_f0,
                    feat_f1,
                )
            )

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_c0.shape[2:] * self.local_resolution,
                "hw1_f": feat_c1.shape[2:] * self.local_resolution,
            }
        )
        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")
        feat_f0 = rearrange(feat_f0, "n c h w -> n (h w) c")
        feat_f1 = rearrange(feat_f1, "n c h w -> n (h w) c")

        # detect NaN during mixed precision training
        if self.config["mp"] and (
            torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))
        ):
            detect_NaN(feat_c0, feat_c1)

        # 3. Coarse-Level Matching
        conf_matrix = self.coarse_matching(
            feat_c0,
            feat_c1,
            data,
            feat_r0=feat_r0,
            feat_r1=feat_r1,
            mask_c0=(
                mask_c0.view(mask_c0.size(0), -
                             1) if mask_c0 is not None else mask_c0
            ),
            mask_c1=(
                mask_c1.view(mask_c1.size(0), -
                             1) if mask_c1 is not None else mask_c1
            ),
        )

        if self.deploy and conf_matrix is not None:
            k = self.topk
            row_max_val, row_max_idx = torch.max(conf_matrix, dim=2)
            topk_val, topk_idx = torch.topk(row_max_val, k, dim=1)

            b_ids = (
                torch.arange(conf_matrix.shape[0], device=conf_matrix.device)
                .unsqueeze(1)
                .repeat(1, k)
                .flatten()
            )
            i_ids = topk_idx.flatten()
            j_ids = row_max_idx[b_ids, i_ids].flatten()
            mconf = conf_matrix[b_ids, i_ids, j_ids]

            scale = data["hw0_i"][0] / data["hw0_c"][0]
            scale0 = scale * \
                data["scale0"][b_ids] if "scale0" in data else scale
            scale1 = scale * \
                data["scale1"][b_ids] if "scale1" in data else scale
            mkpts0_c = (
                torch.stack(
                    [
                        i_ids % data["hw0_c"][1],
                        torch.div(i_ids, data["hw0_c"][1],
                                  rounding_mode="floor"),
                    ],
                    dim=1,
                )
                * scale0
            )
            mkpts1_c = (
                torch.stack(
                    [
                        j_ids % data["hw1_c"][1],
                        torch.div(j_ids, data["hw1_c"][1],
                                  rounding_mode="floor"),
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

        if self.deploy:
            # Both coarse matching paths return fixed-size match tensors in ``data``.
            mkpts0_c = data["mkpts0_c"]
            mkpts1_c = data["mkpts1_c"]
            mconf = data["mconf"]

        self._prefilter_fine_candidates(data)

        # 4. Fine-Level Matching
        K0 = data["i_ids"].shape[0] // data["bs"]
        K1 = data["j_ids"].shape[0] // data["bs"]
        feat_f0 = feat_f0[data["b_ids"], data["i_ids"]
                          ].reshape(data["bs"], K0, -1)
        feat_f1 = feat_f1[data["b_ids"], data["j_ids"]
                          ].reshape(data["bs"], K1, -1)
        feat_c0 = feat_c0[data["b_ids"], data["i_ids"]
                          ].reshape(data["bs"], K0, -1)
        feat_c1 = feat_c1[data["b_ids"], data["j_ids"]
                          ].reshape(data["bs"], K1, -1)

        optimize_fine_layout = not self.training and not torch.is_grad_enabled()
        if optimize_fine_layout:
            # Combine corresponding descriptors before packing the two directions.
            combined0 = feat_f0 + feat_c0
            combined1 = feat_f1 + feat_c1
            query = torch.cat([combined0, combined1], dim=1)
            reference = torch.cat([combined1, combined0], dim=1)
            offset, score = self.fine_matching.forward_precombined(
                query, reference, data
            )
        else:
            offset, score = self.fine_matching(
                torch.cat([feat_f0, feat_f1], dim=1),
                torch.cat([feat_f1, feat_f0], dim=1),
                torch.cat([feat_c0, feat_c1], dim=1),
                torch.cat([feat_c1, feat_c0], dim=1),
                data,
            )

        if self.deploy:
            fine_offset01, fine_offset10 = offset.chunk(2)
            fine_score01, fine_score10 = score.unsqueeze(dim=1).chunk(2)
            return torch.cat(
                [
                    mkpts0_c, mkpts1_c, fine_offset01, fine_offset10,
                    fine_score01, fine_score10, mconf.unsqueeze(dim=1),
                ],
                dim=1,
            )  # [K, 11]

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = rename_state_dict_prefix(state_dict, "matcher", "ultramatch")
        state_dict = rename_state_dict_prefix(state_dict, "ultramatch", "")
        return super().load_state_dict(state_dict, *args, **kwargs)
