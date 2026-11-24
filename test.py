import lightning.pytorch as pl
import argparse
import pprint
import os
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_ultramatch import UltraMatchLightning

import torch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_CONFIG = os.path.join(
    PROJECT_DIR, "configs/data/megadepth_test_1500.py"
)
DEFAULT_MODEL_CONFIG = os.path.join(PROJECT_DIR, "configs/ultramatch/outdoor/ultramatch.py")
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_DIR, "weights/outdoor.ckpt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data_cfg_path",
        nargs="?",
        type=str,
        default=DEFAULT_DATA_CONFIG,
        help="data config path",
    )
    parser.add_argument(
        "main_cfg_path",
        nargs="?",
        type=str,
        default=DEFAULT_MODEL_CONFIG,
        help="main config path",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="ddp")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="path to the pretrained UltraMatch checkpoint",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="if set, the matching results will be dump to dump_dir",
    )
    parser.add_argument("--benchmark", default=True)
    parser.add_argument(
        "--profiler_name",
        type=str,
        default="inference",
        help="options: [inference, pytorch], or leave it unset",
    )
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument(
        "--deter",
        dest="deter",
        action="store_true",
        help="use deterministic mode for testing",
    )
    deterministic.add_argument(
        "--no_deter",
        dest="deter",
        action="store_false",
        help="disable deterministic mode for testing",
    )
    parser.set_defaults(deter=True)
    parser.add_argument(
        "--pixel_thr", type=float, default=None, help="modify the RANSAC threshold."
    )
    parser.add_argument(
        "--ransac", type=str, default=None, help="modify the RANSAC method"
    )
    parser.add_argument(
        "--W", type=int, default=None, help="image width"
    )
    parser.add_argument(
        "--H", type=int, default=None, help="image height"
    )
    parser.add_argument(
        "--coarse_mode",
        type=str,
        choices=["full", "block_routed", "block_sparse"],
    )
    parser.add_argument("--route_topk", type=int)
    parser.add_argument("--block_halo", type=int)
    parser.add_argument("--mconf_thr", type=float, default=None)
    reparameterization = parser.add_mutually_exclusive_group()
    reparameterization.add_argument(
        "--reparameterize",
        dest="reparameterize",
        action="store_true",
        help="fold Conv-BN pairs after loading the inference checkpoint",
    )
    reparameterization.add_argument(
        "--no_reparameterize",
        dest="reparameterize",
        action="store_false",
        help="disable inference Conv-BN/Rep reparameterization",
    )
    parser.set_defaults(reparameterize=None)
    fine_prefilter = parser.add_mutually_exclusive_group()
    fine_prefilter.add_argument(
        "--prefilter_fine",
        dest="prefilter_fine",
        action="store_true",
        help="prepare a fixed-shape coarse-confidence mask before the fine head",
    )
    fine_prefilter.add_argument(
        "--no_prefilter_fine",
        dest="prefilter_fine",
        action="store_false",
        help="disable coarse-confidence prefiltering before the fine head",
    )
    parser.set_defaults(prefilter_fine=None)
    parser.add_argument(
        "--prefilter_strategy",
        choices=["mask", "dynamic", "bucket", "fixed"],
    )
    parser.add_argument("--prefilter_fixed_k", type=int, default=None)
    parser.add_argument(
        "--prefilter_buckets", type=int, nargs="+", default=None
    )
    parser.add_argument(
        "--no_cudnn_benchmark",
        action="store_true",
        help="disable cuDNN algorithm benchmarking for dynamic input shapes",
    )
    parser.add_argument(
        "--feature_amp",
        choices=["off", "bf16", "fp16"],
        help=(
            "override feature precision for the backbone, CIM and 1/32 attention. "
            "BF16 falls back to FP32 on unsupported GPUs"
        ),
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="use channels-last layout for the Conv2d feature path",
    )
    parser.add_argument(
        "--matmul_precision",
        choices=["highest", "high"],
        help="'high' enables TF32 matmuls on supported NVIDIA GPUs",
    )
    attention_fusion = parser.add_mutually_exclusive_group()
    attention_fusion.add_argument(
        "--fuse_attention_projections",
        dest="fuse_attention_projections",
        action="store_true",
        help="fuse self QKV/cross KV projections and batch two-image self attention",
    )
    attention_fusion.add_argument(
        "--no_fuse_attention_projections",
        dest="fuse_attention_projections",
        action="store_false",
        help="disable inference attention projection fusion",
    )
    parser.set_defaults(fuse_attention_projections=None)
    parser.add_argument(
        "--sigma_selection",
        choices=["on", "off"],
        default=None,
        help="override UltraMatch Fine bidirectional sigma selection for evaluation",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=float,
        default=1.0,
        help="fraction of test batches to run (1.0 runs MegaDepth-1500 in full)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pprint.pprint(vars(args))

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    if args.deter:
        # Deterministic CUDA evaluation also requires a fixed cuBLAS workspace.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    pl.seed_everything(config.TRAINER.SEED, workers=args.deter)  # reproducibility

    if args.pixel_thr is not None:
        config.TRAINER.RANSAC_PIXEL_THR = args.pixel_thr

    if args.ransac is not None:
        config.TRAINER.POSE_ESTIMATION_METHOD = args.ransac

    if (args.W is None) != (args.H is None):
        raise ValueError("Provide both --W and --H to override the test resolution")
    if args.W is not None and args.H is not None:
        config.ULTRAMATCH.TEST_RES_W = args.W
        config.ULTRAMATCH.TEST_RES_H = args.H
        config.ULTRAMATCH.COARSE.TOPK = int(args.W / 8 * args.H / 8 * 0.35)

    if args.coarse_mode is not None:
        config.ULTRAMATCH.COARSE.MODE = args.coarse_mode
    if args.route_topk is not None:
        config.ULTRAMATCH.COARSE.ROUTE_TOPK = args.route_topk
    if args.block_halo is not None:
        config.ULTRAMATCH.COARSE.BLOCK_HALO = args.block_halo
    if args.mconf_thr is not None:
        config.ULTRAMATCH.COARSE.MCONF_THR = args.mconf_thr
    if args.reparameterize is not None:
        config.ULTRAMATCH.REPARAMETERIZE = args.reparameterize
    if args.prefilter_fine is not None:
        config.ULTRAMATCH.COARSE.PREFILTER_FINE = args.prefilter_fine
    if args.prefilter_strategy is not None:
        config.ULTRAMATCH.COARSE.PREFILTER_STRATEGY = args.prefilter_strategy
    if args.prefilter_fixed_k is not None:
        config.ULTRAMATCH.COARSE.PREFILTER_FIXED_K = args.prefilter_fixed_k
    if args.prefilter_buckets is not None:
        config.ULTRAMATCH.COARSE.PREFILTER_BUCKETS = args.prefilter_buckets
    if args.feature_amp is not None:
        config.ULTRAMATCH.FEATURE_AMP = args.feature_amp
    if args.channels_last:
        config.ULTRAMATCH.CHANNELS_LAST = True
    if args.matmul_precision is not None:
        config.ULTRAMATCH.MATMUL_PRECISION = args.matmul_precision
    if args.fuse_attention_projections is not None:
        config.ULTRAMATCH.FUSE_ATTENTION_PROJECTIONS = args.fuse_attention_projections
    if args.sigma_selection is not None:
        config.ULTRAMATCH.FINE.SIGMA_SELECTION = args.sigma_selection == "on"
    torch.set_float32_matmul_precision(config.ULTRAMATCH.MATMUL_PRECISION)

    loguru_logger.info(f"Args and config initialized!")

    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"UltraMatch checkpoint not found: {args.ckpt_path}")

    profiler = build_profiler(args.profiler_name)
    model = UltraMatchLightning(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        dump_dir=args.dump_dir,
    )
    loguru_logger.info("UltraMatch model initialized")

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp",
        num_nodes=args.num_nodes,
        benchmark=False if args.deter else not args.no_cudnn_benchmark,
        deterministic=args.deter,
        logger=False,
        use_distributed_sampler=False,
        profiler=profiler,
        limit_test_batches=args.limit_test_batches,
    )

    loguru_logger.info(f"Start testing!")

    trainer.test(model, datamodule=data_module, verbose=False)
