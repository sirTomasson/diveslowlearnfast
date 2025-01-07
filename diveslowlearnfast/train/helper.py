import pytorchvideo
import torch
from pytorchvideo.transforms import Div255, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.transforms import ToTensor4D, Permute, RandomRotateVideo


def get_aug_paras(cfg: Config):
    if not cfg.RAND_AUGMENT.ENABLED:
        return None

    return {
        'num_layers': cfg.RAND_AUGMENT.NUM_LAYERS,
        'magnitude': cfg.RAND_AUGMENT.MAGNITUDE,
        'prob': cfg.RAND_AUGMENT.PROB,
    }


def get_aug_type(cfg: Config):
    if cfg.RAND_AUGMENT.ENABLED:
        return 'randaug'

    return 'default'


def get_train_transform(cfg: Config, crop_size=None):
    crop_size = cfg.DATA.TRAIN_CROP_SIZE if crop_size is None else crop_size

    transformations = [
        ToTensor4D(),
        Permute(3, 0, 1, 2),  # From T x H X W x 3 -> 3 x T x H x W
        Div255(),
        pytorchvideo.transforms.create_video_transform(
            mode='train',
            num_samples=cfg.DATA.NUM_FRAMES,
            video_std=cfg.DATA.MEAN,
            video_mean=cfg.DATA.STD,
            convert_to_float=False,
            crop_size=crop_size,
            aug_type=get_aug_type(cfg),
            aug_paras=get_aug_paras(cfg),
            horizontal_flip_prob=0.5,
        )
    ]
    if cfg.RANDOM_ROTATE.ENABLED:
        angle = cfg.RANDOM_ROTATE.MAX_DEGREE
        transformations.extend([
            Permute(1, 0, 2, 3),  # From 3 x T x H X W -> T x 3 x H x W; because RandAug expects this shape
            RandomRotateVideo(-angle, angle),
            Permute(1, 0, 2, 3),  # From T x 3 x H X W -> 3 x T x H x W
        ])
    return Compose(transformations)


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


def get_test_objects(cfg, include_labels=None):
    test_transform = get_test_transform(cfg)

    test_dataset = Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        dataset_type='test',
        transform_fn=test_transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD,
        multi_thread_decode=cfg.DATA.MULTI_THREAD_DECODE,
        include_labels=include_labels,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=False,
    )

    return test_loader


def get_include_labels(cfg):
    if cfg.DATA.THRESHOLD < 0:
        return None

    return Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        dataset_type='train',
        threshold=cfg.DATA.THRESHOLD,
    ).labels


def get_train_objects(cfg, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    train_transform = get_train_transform(cfg)

    train_dataset = Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        dataset_type='train',
        transform_fn=train_transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD,
        temporal_random_jitter=cfg.DATA.TEMPORAL_RANDOM_JITTER,
        temporal_random_offset=cfg.DATA.TEMPORAL_RANDOM_OFFSET,
        multi_thread_decode=cfg.DATA.MULTI_THREAD_DECODE,
        threshold=cfg.DATA.THRESHOLD,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=True,
    )

    return criterion, optimiser, train_loader, train_dataset
