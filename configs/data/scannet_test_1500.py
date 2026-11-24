from configs.data.base import get_data_cfg_defaults

cfg = get_data_cfg_defaults()

TEST_BASE_PATH = "assets/scannet_test_1500"

cfg.DATASET.TEST_DATA_SOURCE = "ScanNet"
cfg.DATASET.TEST_DATA_ROOT = "data/scannet/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/scannet_test.txt"
cfg.DATASET.TEST_INTRINSIC_PATH = f"{TEST_BASE_PATH}/intrinsics.npz"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# Evaluate model trained on MegaDepth
cfg.ULTRAMATCH.TRAIN_RES_H = 832
cfg.ULTRAMATCH.TRAIN_RES_W = 832
cfg.ULTRAMATCH.TEST_RES_H = 480
cfg.ULTRAMATCH.TEST_RES_W = 640

cfg.ULTRAMATCH.NECK.NPE = [
    cfg.ULTRAMATCH.TRAIN_RES_H,
    cfg.ULTRAMATCH.TRAIN_RES_W,
    cfg.ULTRAMATCH.TEST_RES_H,
    cfg.ULTRAMATCH.TEST_RES_W,
]
