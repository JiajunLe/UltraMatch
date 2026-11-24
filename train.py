import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_ultramatch import UltraMatchLightning

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="ddp")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to an initial UltraMatch checkpoint",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="full training-state checkpoint used to resume an interrupted run",
    )
    parser.add_argument(
        "--recovery_ckpt_every_n_train_steps",
        type=int,
        default=1000,
        help="save one rotating full-state recovery checkpoint every N train steps; 0 disables it",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--num_sanity_val_steps", type=int, default=10)
    parser.add_argument("--benchmark", default=True)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--limit_train_batches", type=int, default=-1)
    parser.add_argument("--limit_val_batches", type=int, default=-1)
    parser.add_argument("--split_data_idx", type=int, default=0)
    parser.add_argument(
        "--parallel_load_data",
        default=False,
        action="store_true",
        help="load datasets in with multiple processes.",
    )
    parser.add_argument(
        "--disable_ckpt",
        action="store_true",
        help="disable checkpoint saving (useful for debugging).",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=10,
        help="number of monitored epoch checkpoints to retain; -1 saves all",
    )
    parser.add_argument(
        "--profiler_name",
        type=str,
        default=None,
        help="options: [inference, pytorch], or leave it unset",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(
        config.TRAINER.WARMUP_STEP / _scaling)

    profiler = build_profiler(args.profiler_name)
    model = UltraMatchLightning(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"UltraMatch LightningModule initialized!")

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"UltraMatch DataModule initialized!")

    logger = TensorBoardLogger(
        save_dir="logs/tb_logs", name=args.exp_name, default_hp_metric=False
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    ckpt_callback = ModelCheckpoint(
        monitor="auc@10",
        verbose=True,
        save_top_k=args.save_top_k,
        mode="max",
        save_last=True,
        save_weights_only=True,
        dirpath=str(ckpt_dir),
        filename="{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
        if args.recovery_ckpt_every_n_train_steps > 0:
            callbacks.append(
                ModelCheckpoint(
                    monitor=None,
                    save_top_k=1,
                    save_last=False,
                    save_weights_only=False,
                    every_n_train_steps=(
                        args.recovery_ckpt_every_n_train_steps
                    ),
                    dirpath=str(ckpt_dir / "recovery"),
                    filename="recovery-{epoch}-{step}",
                )
            )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp",
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        limit_train_batches=(
            1.0 if args.limit_train_batches < 0 else args.limit_train_batches
        ),
        limit_val_batches=(
            1.0 if args.limit_val_batches < 0 else args.limit_val_batches
        ),
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        benchmark=True,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        enable_checkpointing=not args.disable_ckpt,
        logger=logger,
        sync_batchnorm=config.TRAINER.SYNC_BATCHNORM,
        use_distributed_sampler=False,
        profiler=profiler,
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
