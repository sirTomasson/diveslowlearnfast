import pytorchvideo
import torch
from pytorchvideo.transforms import Div255, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.transforms import ToTensor4D, Permute


def get_train_transform(cfg: Config, crop_size=None):
    crop_size = cfg.DATA.TRAIN_CROP_SIZE if crop_size is None else crop_size
    return Compose([
        ToTensor4D(),
        Permute(3, 0, 1, 2), # From T x H X W x 3 -> 3 x T x H x W
        Div255(),
        pytorchvideo.transforms.create_video_transform(
            mode='train',
            num_samples=cfg.DATA.NUM_FRAMES,
            video_std=cfg.DATA.MEAN,
            video_mean=cfg.DATA.STD,
            convert_to_float=False,
            crop_size=crop_size,
            horizontal_flip_prob=0.5,
            random_resized_crop_paras={'scale': (1.0, 1.0), 'aspect_ratio': (1.0, 1.0)}
        ),
        Permute(1, 0, 2, 3), # From 3 x T x H X W -> T x 3 x H x W; because RandAug expects this shape
        RandAugment(prob=0.5, sampling_type='gaussian'),
        Permute(1, 0, 2, 3), # From T x 3 x H X W -> 3 x T x H x W
    ])

def get_test_transform(cfg: Config):
    return Compose([
        ToTensor4D(),
        Permute(3, 0, 1, 2),
        Div255(),
        pytorchvideo.transforms.create_video_transform(
            mode='test',
            num_samples=cfg.DATA.NUM_FRAMES,
            video_std=cfg.DATA.MEAN,
            video_mean=cfg.DATA.STD,
            convert_to_float=False,
            crop_size=cfg.DATA.TEST_CROP_SIZE,
        ),
    ])

def get_train_objects(cfg, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    train_transform = get_train_transform(cfg)

    test_transform = get_test_transform(cfg)

    train_dataset = Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        dataset_type='train',
        transform_fn=train_transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD,
        temporal_random_jitter=cfg.DATA.TEMPORAL_RANDOM_JITTER,
        temporal_random_offset=cfg.DATA.TEMPORAL_RANDOM_OFFSET,
        multi_thread_decode=cfg.DATA.MULTI_THREAD_DECODE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=True,
    )

    test_dataset = Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        dataset_type='test',
        transform_fn=test_transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD,
        multi_thread_decode=cfg.DATA.MULTI_THREAD_DECODE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=False,
    )
    return criterion, optimiser, train_loader, test_loader