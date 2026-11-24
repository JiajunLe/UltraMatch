from yacs.config import CfgNode as CN

_CN = CN()

##############  ↓  UltraMatch Pipeline  ↓  ##############
_CN.ULTRAMATCH = CN()
_CN.ULTRAMATCH.TRAIN_RES_H = 832
_CN.ULTRAMATCH.TRAIN_RES_W = 832
_CN.ULTRAMATCH.TEST_RES_H = 832
_CN.ULTRAMATCH.TEST_RES_W = 832
_CN.ULTRAMATCH.LOCAL_RESOLUTION = 8  # coarse matching at 1/8, window_size 8x8 in fine_level
_CN.ULTRAMATCH.MP = False
_CN.ULTRAMATCH.HALF = False
_CN.ULTRAMATCH.DEPLOY = False # export onnx model
_CN.ULTRAMATCH.EVAL_TIMES = 1
_CN.ULTRAMATCH.REPARAMETERIZE = False  # fold Conv-BN only in a final eval model
_CN.ULTRAMATCH.FEATURE_AMP = "off"  # off, bf16, or fp16; inference feature path only
_CN.ULTRAMATCH.CHANNELS_LAST = False  # Conv2d feature path only
_CN.ULTRAMATCH.MATMUL_PRECISION = "highest"  # highest or high (TF32 on Ampere)
_CN.ULTRAMATCH.FUSE_ATTENTION_PROJECTIONS = False

# Backbone
_CN.ULTRAMATCH.BACKBONE = CN()
_CN.ULTRAMATCH.BACKBONE.BLOCK_DIMS = [32, 64, 128, 256, 256]  # 1/2 -> 1/32
# Enable multi-branch convolutions in the selected stages during training.
_CN.ULTRAMATCH.BACKBONE.REP_ENABLED = False
_CN.ULTRAMATCH.BACKBONE.REP_STAGES = []
_CN.ULTRAMATCH.BACKBONE.REP_FINE = False
_CN.ULTRAMATCH.BACKBONE.REP_USE_1X1 = True
_CN.ULTRAMATCH.BACKBONE.REP_USE_DIRECTIONAL = True
_CN.ULTRAMATCH.BACKBONE.REP_AUXILIARY_BN_INIT = 0.0
# False preserves the original independent Conv-BN branch graph and its
# checkpoint keys. True sums branch convolutions before one shared BN.
_CN.ULTRAMATCH.BACKBONE.REP_SHARED_BN = False

# Feature interaction
_CN.ULTRAMATCH.NECK = CN()
_CN.ULTRAMATCH.NECK.D_MODEL = 256
_CN.ULTRAMATCH.NECK.NHEAD = 8
_CN.ULTRAMATCH.NECK.LAYER_NAMES = ["self", "cross"] * 2
_CN.ULTRAMATCH.NECK.AGG_SIZE0 = 1
_CN.ULTRAMATCH.NECK.AGG_SIZE1 = 1
_CN.ULTRAMATCH.NECK.ROPE = True
_CN.ULTRAMATCH.NECK.NPE = None
_CN.ULTRAMATCH.NECK.ATTENTION_IMPL = "legacy"  # legacy or fused cosine SDPA
# Training-time directional branches for CIM's two depthwise 3x3 convolutions.
# They fuse back to the original depthwise 3x3 inference operators.
_CN.ULTRAMATCH.NECK.REP_DW = False

# Coarse matching
_CN.ULTRAMATCH.COARSE = CN()
_CN.ULTRAMATCH.COARSE.MCONF_THR = 0.2
_CN.ULTRAMATCH.COARSE.BORDER_RM = 0
_CN.ULTRAMATCH.COARSE.DSMAX_TEMPERATURE = 0.1
_CN.ULTRAMATCH.COARSE.TRAIN_PAD_NUM = 32  # training tricks: avoid DDP deadlock
_CN.ULTRAMATCH.COARSE.TOPK = 2048
_CN.ULTRAMATCH.COARSE.DS_OPT = True
_CN.ULTRAMATCH.COARSE.MODE = "full"  # full, block_routed, block_sparse
_CN.ULTRAMATCH.COARSE.BLOCK_SIZE = 4  # 1/32 router blocks over the 1/8 matching grid
_CN.ULTRAMATCH.COARSE.ROUTE_TOPK = 8  # number of target blocks selected per source block
_CN.ULTRAMATCH.COARSE.BLOCK_HALO = 1  # extra 1/8 cells around each routed target block
_CN.ULTRAMATCH.COARSE.PREFILTER_FINE = False  # fixed-shape confidence mask before fine
_CN.ULTRAMATCH.COARSE.PREFILTER_STRATEGY = "mask"  # mask, dynamic, bucket, fixed
_CN.ULTRAMATCH.COARSE.PREFILTER_BUCKETS = [2048, 3072, 4096, 5120, 6144, 7296]
_CN.ULTRAMATCH.COARSE.PREFILTER_FIXED_K = 6144
_CN.ULTRAMATCH.COARSE.FUSED_SPARSE_SELECTION = False

# Trainable block router. Disabled by default to preserve old checkpoints and
# the original full-matching behavior.
_CN.ULTRAMATCH.ROUTER = CN()
_CN.ULTRAMATCH.ROUTER.ENABLED = False
_CN.ULTRAMATCH.ROUTER.DIM = 64
_CN.ULTRAMATCH.ROUTER.TEMPERATURE = 0.1
_CN.ULTRAMATCH.ROUTER.LOSS_WEIGHT = 1.0
_CN.ULTRAMATCH.ROUTER.TRAIN_ONLY = False
_CN.ULTRAMATCH.ROUTER.FREEZE_FEATURES = True
_CN.ULTRAMATCH.ROUTER.BLOCK_CHUNK_SIZE = 128
_CN.ULTRAMATCH.ROUTER.EVAL_ROUTED = False
_CN.ULTRAMATCH.ROUTER.TRAIN_ROUTED = False
_CN.ULTRAMATCH.ROUTER.SPARSE_ONLY = False
_CN.ULTRAMATCH.ROUTER.GT_INJECT_EPOCHS = 0.0
_CN.ULTRAMATCH.ROUTER.ROUTED_COARSE_WEIGHT = 1.0
_CN.ULTRAMATCH.ROUTER.SCORE_PRIOR_ENABLED = False
_CN.ULTRAMATCH.ROUTER.SCORE_PRIOR_INIT = 0.0
_CN.ULTRAMATCH.ROUTER.SCORE_PRIOR_MAX = 1.0
# Route sizes R to cycle through during training. An empty list uses ROUTE_TOPK.
_CN.ULTRAMATCH.ROUTER.TRAIN_SIZES = []
# Zero weights disable the auxiliary geometry losses.
_CN.ULTRAMATCH.ROUTER.COVERAGE_LOSS_WEIGHT = 0.0
_CN.ULTRAMATCH.ROUTER.RANK_LOSS_WEIGHT = 0.0
_CN.ULTRAMATCH.ROUTER.RANK_MARGIN = 0.5

# Shared Parameter Tiny Fine Matching
_CN.ULTRAMATCH.FINE = CN()
_CN.ULTRAMATCH.FINE.MATCHER_TYPE = "tiny_fine_matching"
_CN.ULTRAMATCH.FINE.DROPRATE = None
_CN.ULTRAMATCH.FINE.COORD_LENGTH = 16
_CN.ULTRAMATCH.FINE.BI_DIRECTIONAL_REFINE = True
_CN.ULTRAMATCH.FINE.SIGMA_THR = 0.0
_CN.ULTRAMATCH.FINE.SIGMA_SELECTION = True
_CN.ULTRAMATCH.FINE.DESCRIPTOR_HIDDEN_DIM = 256
_CN.ULTRAMATCH.FINE.PAIR_HIDDEN_DIM = 512
_CN.ULTRAMATCH.FINE.AXIS_HIDDEN_DIM = 256

# Coarse loss
_CN.ULTRAMATCH.LOSS = CN()
_CN.ULTRAMATCH.LOSS.COARSE_TYPE = "focal"
_CN.ULTRAMATCH.LOSS.COARSE_WEIGHT = 1.0
_CN.ULTRAMATCH.LOSS.SPARSE_SPVS = True

# Focal loss parameters
_CN.ULTRAMATCH.LOSS.FOCAL_ALPHA = 0.25
_CN.ULTRAMATCH.LOSS.FOCAL_GAMMA = 2.0
_CN.ULTRAMATCH.LOSS.POS_WEIGHT = 1.0
_CN.ULTRAMATCH.LOSS.NEG_WEIGHT = 1.0


# Fine loss
_CN.ULTRAMATCH.LOSS.FINE_TYPE = "rle"
_CN.ULTRAMATCH.LOSS.FINE_WEIGHT = 0.2
_CN.ULTRAMATCH.LOSS.Q_DISTRIBUTION = "laplace"  # options: ['laplace', 'gaussian']


##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = (
    None  # None if val data from all scenes are bundled into a single npz file
)
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = (
    None  # None if test data from all scenes are bundled into a single npz file
)
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = (
    0.4  # discard data with overlap_score < min_overlap_score
)
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None

# MegaDepth options
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8


##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.SYNC_BATCHNORM = True
_CN.TRAINER.CANONICAL_BS = 4 * 8
_CN.TRAINER.CANONICAL_LR = 2e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.0  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = "linear"  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.0
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
# [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER = "MultiStepLR"
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"  # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
# ELR: ExponentialLR, this value for 'step' interval
_CN.TRAINER.ELR_GAMMA = 0.999992

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = False
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32
_CN.TRAINER.PLOT_MODE = "evaluation"
_CN.TRAINER.PLOT_MATCHES_ALPHA = "dynamic"

# geometric metrics and pose solver
# recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.EPI_ERR_THR = 5e-4
_CN.TRAINER.POSE_GEO_MODEL = "E"  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = "RANSAC"  # [RANSAC, LO-RANSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = "scene_balance"
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
# whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True
# after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# Seed for model initialization and data sampling.
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Return an independent copy of the default configuration."""
    return _CN.clone()
