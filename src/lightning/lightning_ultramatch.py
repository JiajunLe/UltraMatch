from collections import defaultdict
import os
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import lightning.pytorch as pl
from matplotlib import pyplot as plt

from src.ultramatch import UltraMatch
from src.ultramatch.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
)
from src.losses.ultramatch_loss import UltraMatchLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from src.utils.plotting import make_matching_figures
from src.utils.checkpoint import rename_state_dict_prefix
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class UltraMatchLightning(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)

        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        self.ultramatch = UltraMatch(config=_config["ultramatch"])
        self.loss = UltraMatchLoss(_config)
        self.train_route_sizes = tuple(
            int(k) for k in config.ULTRAMATCH.ROUTER.TRAIN_SIZES
        )

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")[
                "state_dict"]
            router_enabled = config.ULTRAMATCH.ROUTER.ENABLED
            incompatible = self.ultramatch.load_state_dict(
                state_dict, strict=not router_enabled
            )
            if router_enabled:
                unexpected = list(incompatible.unexpected_keys)
                non_router_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not key.startswith("coarse_matching.router.")
                ]
                if unexpected or non_router_missing:
                    raise RuntimeError(
                        "Checkpoint mismatch outside the new router: "
                        f"missing={non_router_missing}, unexpected={unexpected}"
                    )
                if incompatible.missing_keys:
                    logger.info(
                        "Initialize new router parameters not present in the "
                        "pretrained UltraMatch checkpoint"
                    )
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.warmup = False
        self.reparameter = False
        self.attention_reparameter = False
        self.channels_last = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.total_timed_pairs = 0
        self.dump_dir = dump_dir

        # outputs
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Recovery checkpoints saved before the package rename use "matcher.".
        state_dict = rename_state_dict_prefix(state_dict, "matcher", "ultramatch")
        return super().load_state_dict(state_dict, *args, **kwargs)

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("UltraMatch"):
            with torch.autocast(enabled=self.config.ULTRAMATCH.MP, device_type="cuda"):
                self.ultramatch(batch)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.ULTRAMATCH.MP, device_type="cuda"):
                self.loss(batch)

    def _compute_metrics(self, batch):
        # compute epi_errs for each match
        compute_symmetrical_epipolar_errors(batch)

        compute_pose_errors(
            batch, self.config
        )  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
            "epi_errs": [
                (batch["epi_errs"].reshape(-1, 1))[batch["m_bids"] == b]
                .reshape(-1)
                .cpu()
                .numpy()
                for b in range(bs)
            ],
            "R_errs": batch["R_errs"],
            "t_errs": batch["t_errs"],
            "inliers": batch["inliers"],
            "num_matches": [batch["mconf"].shape[0]],  # batch size = 1 only
        }
        ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        if self.train_route_sizes:
            # All DDP ranks cycle through the same route sizes in the same order.
            size_index = (
                int(self.global_step) + int(self.config.TRAINER.SEED)
            ) % len(self.train_route_sizes)
            batch["train_route_size"] = self.train_route_sizes[
                size_index
            ]
        num_batches = max(int(self.trainer.num_training_batches), 1)
        batch["train_epoch_progress"] = (
            float(self.current_epoch) + float(batch_idx) / num_batches
        )
        self._trainval_inference(batch)

        if not torch.isfinite(batch["loss"]):
            components = {
                key: float(value)
                for key, value in batch["loss_scalars"].items()
            }
            raise FloatingPointError(
                f"Non-finite training loss at step {self.global_step}: "
                f"{components}"
            )

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(
                    f"train/{k}", v, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(
                    batch
                )  # compute epi_errs for each match
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.add_figure(
                        f"train_match/{k}", v, self.global_step
                    )

        out = {"loss": batch["loss"]}
        self.log("loss", batch["loss"], prog_bar=True, rank_zero_only=True)
        if "route_recall01" in batch:
            route_recall = 0.5 * (
                batch["route_recall01"] + batch["route_recall10"]
            )
            self.log(
                f"route_recall@{self.config.ULTRAMATCH.COARSE.ROUTE_TOPK}",
                route_recall,
                prog_bar=True,
                sync_dist=True,
            )
        if "routed_gt_recall" in batch:
            self.log(
                "routed_gt_recall",
                batch["routed_gt_recall"],
                prog_bar=True,
                sync_dist=True,
            )
        if "route_size_used" in batch:
            self.log(
                "route_size",
                batch["route_size_used"],
                prog_bar=True,
                sync_dist=True,
            )
        for route_size in self.train_route_sizes:
            exact01 = batch.get(f"route_recall01_r{route_size}")
            exact10 = batch.get(f"route_recall10_r{route_size}")
            halo01 = batch.get(f"route_halo_recall01_r{route_size}")
            halo10 = batch.get(f"route_halo_recall10_r{route_size}")
            if exact01 is not None and exact10 is not None:
                self.log(
                    f"route_exact@{route_size}",
                    0.5 * (exact01 + exact10),
                    prog_bar=False,
                    sync_dist=True,
                )
            if halo01 is not None and halo10 is not None:
                self.log(
                    f"route_halo@{route_size}",
                    0.5 * (halo01 + halo10),
                    prog_bar=route_size == min(self.train_route_sizes),
                    sync_dist=True,
                )
        if "route_score_prior_weight" in batch:
            self.log(
                "route_prior_weight",
                batch["route_score_prior_weight"],
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "route_prior_active",
                batch["route_score_prior_active"],
                prog_bar=False,
                sync_dist=True,
            )

        return out

    def on_after_backward(self) -> None:
        if os.environ.get("ULTRAMATCH_DEBUG_GRADS") == "1":
            for name, parameter in self.named_parameters():
                if parameter.grad is not None and not torch.isfinite(
                    parameter.grad
                ).all():
                    finite = parameter.grad[torch.isfinite(parameter.grad)]
                    max_abs = (
                        finite.abs().max().item() if finite.numel() else float("nan")
                    )
                    logger.error(
                        f"Non-finite gradient: {name}, "
                        f"finite={finite.numel()}/{parameter.grad.numel()}, "
                        f"finite_max_abs={max_abs}"
                    )
        return super().on_after_backward()

    def on_train_start(self) -> None:
        if os.environ.get("ULTRAMATCH_DETECT_ANOMALY") == "1":
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Autograd anomaly detection enabled for debugging")
        return super().on_train_start()

    def on_train_epoch_start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_end(self):
        if torch.cuda.is_available():
            peak_mib = torch.cuda.max_memory_allocated(self.device) / 1024**2
            logger.info(f"Training peak allocated memory: {peak_mib:.2f} MiB")

    def on_test_start(self):
        self.total_timed_pairs = 0
        self.total_ms = 0
        if self.config.ULTRAMATCH.REPARAMETERIZE and not self.reparameter:
            self.ultramatch.eval().reparameterize_for_inference()
            self.reparameter = True
            logger.info("Fused Conv-BN pairs for inference")
        if (
            self.config.ULTRAMATCH.FUSE_ATTENTION_PROJECTIONS
            and not self.attention_reparameter
        ):
            self.ultramatch.eval().fuse_attention_for_inference()
            self.attention_reparameter = True
            logger.info("Fused Transformer QKV/KV projections for inference")
        if self.config.ULTRAMATCH.CHANNELS_LAST and not self.channels_last:
            self.ultramatch.enable_channels_last_for_inference()
            self.channels_last = True
            logger.info("Enabled channels-last Conv2d feature path")

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        if self.config.ULTRAMATCH.ROUTER.EVAL_ROUTED:
            actual_mode = batch.get("coarse_mode")
            expected_mode = self.config.ULTRAMATCH.COARSE.MODE
            if actual_mode != expected_mode:
                raise RuntimeError(
                    "Router validation must use deployed coarse mode: "
                    f"expected={expected_mode}, actual={actual_mode}"
                )

        if "route_recall01" in batch:
            self.log(
                f"val_route_recall@{self.config.ULTRAMATCH.COARSE.ROUTE_TOPK}",
                0.5 * (batch["route_recall01"] + batch["route_recall10"]),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        for route_size in self.train_route_sizes:
            exact01 = batch.get(f"route_recall01_r{route_size}")
            exact10 = batch.get(f"route_recall10_r{route_size}")
            halo01 = batch.get(f"route_halo_recall01_r{route_size}")
            halo10 = batch.get(f"route_halo_recall10_r{route_size}")
            if exact01 is not None and exact10 is not None:
                self.log(
                    f"val_route_exact@{route_size}",
                    0.5 * (exact01 + exact10),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            if halo01 is not None and halo10 is not None:
                self.log(
                    f"val_route_halo@{route_size}",
                    0.5 * (halo01 + halo10),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(
            self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if (
            self.config.TRAINER.ENABLE_PLOTTING
            and batch_idx % val_plot_interval == 0
        ):
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        out = {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            "figures": figures,
        }
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        # handle multiple validation sets
        multi_outputs = (
            [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # Record the initial validation sanity check at epoch -1.
            cur_epoch = self.trainer.current_epoch
            if self.trainer.ckpt_path is None and self.trainer.sanity_checking:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {
                k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(
                    flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            # All ranks aggregate metrics for the checkpoint callback.
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(
                    val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(
                    gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(
                        f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch
                    )

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                                fig,
                                cur_epoch,
                                close=True,
                            )
            plt.close("all")

        for thr in [5, 10, 20]:
            self.log(
                f"auc@{thr}",
                torch.tensor(
                    float(np.mean(multi_val_metrics[f"auc@{thr}"])),
                    device=self.device,
                    dtype=torch.float32,
                ),
                sync_dist=True,
            )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        if self.config.ULTRAMATCH.HALF:
            self.ultramatch = self.ultramatch.eval().half()

        # Following EfficientLoFTR
        if not self.warmup:
            if self.config.ULTRAMATCH.HALF:
                for i in range(50):
                    self.ultramatch(batch)
            else:
                with torch.autocast(enabled=self.config.ULTRAMATCH.MP, device_type="cuda"):
                    for i in range(50):
                        self.ultramatch(batch)
            self.warmup = True

        torch.cuda.synchronize()
        if self.config.ULTRAMATCH.HALF:
            self.start_event.record()
            self.ultramatch(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.ULTRAMATCH.MP, device_type="cuda"):
                self.start_event.record()
                self.ultramatch(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        self.total_timed_pairs += int(batch["image0"].shape[0])
        ret_dict, rel_pair_names = self._compute_metrics(batch)

        if self.dump_dir is not None:
            with self.profiler.profile("dump_results"):
                # dump results for further analysis
                keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ["R_errs", "t_errs", "inliers"]:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        self.test_step_outputs.append(ret_dict)
        return ret_dict

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]

        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # dump
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"]
                                 for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        gathered_pair_counts = gather(self.total_timed_pairs)
        gathered_total_ms = gather(self.total_ms)
        if self.trainer.global_rank == 0:
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            total_pairs = sum(gathered_pair_counts)
            mean_total_ms = sum(gathered_total_ms) / max(total_pairs, 1)
            print(
                f"Averaged Matching time over {total_pairs} pairs: "
                f"{mean_total_ms:.2f} ms"
            )
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "ultramatch_pred_eval", dumps)

        self.test_step_outputs.clear()
