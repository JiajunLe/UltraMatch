from configs.data.base import get_data_cfg_defaults

cfg = get_data_cfg_defaults()

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

cfg.ULTRAMATCH.TRAIN_RES_H = 832
cfg.ULTRAMATCH.TRAIN_RES_W = 832
cfg.ULTRAMATCH.TEST_RES_H = 1152
cfg.ULTRAMATCH.TEST_RES_W = 1152

cfg.ULTRAMATCH.NECK.NPE = [
    cfg.ULTRAMATCH.TRAIN_RES_H,
    cfg.ULTRAMATCH.TRAIN_RES_W,
    cfg.ULTRAMATCH.TEST_RES_H,
    cfg.ULTRAMATCH.TEST_RES_W,
]
