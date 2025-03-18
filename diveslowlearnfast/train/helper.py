from typing import Iterator

import pytorchvideo
import torch
import random

from pytorchvideo.transforms import Div255, RandomShortSideScale, Normalize
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, RandomCrop, RandomHorizontalFlip

from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets import Diving48Dataset, Diving48ConfounderDatasetWrapper
from diveslowlearnfast.loss.rrr import DualPathRRRLoss
from diveslowlearnfast.train.stats import Statistics
from diveslowlearnfast.transforms import ToTensor4D, Permute, RandomRotateVideo


def get_batch(loader: Iterator,
              device: torch.device,
              stats: Statistics = None,
              data_requires_grad: bool = False):
    if stats:
        with stats.timer('loader_time'):
            xb, yb, io_times, transform_times, video_ids, masks_slow, masks_fast = next(loader)
    else:
        xb, yb, io_times, transform_times, video_ids, masks_slow, masks_fast = next(loader)

    if stats:
        stats.update(
            io_time=io_times.numpy().mean(),
            transform_time=transform_times.numpy().mean()
        )
    xb = xb.to(device)
    xb.requires_grad = data_requires_grad
    return xb, yb.to(device), video_ids, masks_slow.to(device), masks_fast.to(device)


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
        RandomShortSideScale(
            min_size=256,
            max_size=320,
        ),
        RandomCrop(cfg.DATA.TRAIN_CROP_SIZE),
    ]

    aug_type = get_aug_type(cfg)
    if aug_type == 'randaug':
        transformations.append(
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
        )
    else:
        transformations.extend([
            RandomCrop(cfg.DATA.TRAIN_CROP_SIZE),
            Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
            RandomHorizontalFlip(p=0.5),
        ])

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
        cfg.SLOWFAST.ALPHA,
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
        alpha=cfg.SLOWFAST.ALPHA,
        dataset_type='train',
        threshold=cfg.DATA.THRESHOLD,
    ).labels


def get_train_loader_and_dataset(cfg, video_ids=None):
    train_transform = get_train_transform(cfg)

    # take a random sample of n classes
    include_labels = None
    if cfg.CONFOUNDERS.ENABLED and \
            cfg.CONFOUNDERS.GRID_SIZE > cfg.CONFOUNDERS.NUM_INCLUDE_CLASSES:
        include_labels = random.sample(
            range(cfg.CONFOUNDERS.GRID_SIZE),
            cfg.CONFOUNDERS.NUM_INCLUDE_CLASSES
        )

    if len(cfg.DATA.INCLUDE_LABELS) > 0:
        include_labels = cfg.DATA.INCLUDE_LABELS

    if cfg.EGL.ENABLED:
        return_train_dataset = Diving48Dataset(
            cfg.DATA.DATASET_PATH,
            cfg.DATA.NUM_FRAMES,
            alpha=cfg.SLOWFAST.ALPHA,
            dataset_type='train',
            transform_fn=train_transform,
            use_decord=cfg.DATA_LOADER.USE_DECORD,
            temporal_random_jitter=cfg.DATA.TEMPORAL_RANDOM_JITTER,
            temporal_random_offset=cfg.DATA.TEMPORAL_RANDOM_OFFSET,
            multi_thread_decode=cfg.DATA.MULTI_THREAD_DECODE,
            threshold=cfg.DATA.THRESHOLD,
            use_sampling_ratio=cfg.DATA.USE_SAMPLING_RATIO,
            masks_cache_dir=cfg.EGL.MASKS_CACHE_DIR,
            video_ids=video_ids,
            include_labels=include_labels,
            extend_classes=cfg.DATA.EXTEND_CLASSES,
        )
    else:
        return_train_dataset = Diving48Dataset(
            cfg.DATA.DATASET_PATH,
            cfg.DATA.NUM_FRAMES,
            alpha=cfg.SLOWFAST.ALPHA,
            dataset_type='train',
            transform_fn=train_transform,
            use_decord=cfg.DATA_LOADER.USE_DECORD,
            temporal_random_jitter=cfg.DATA.TEMPORAL_RANDOM_JITTER,
            temporal_random_offset=cfg.DATA.TEMPORAL_RANDOM_OFFSET,
            multi_thread_decode=cfg.DATA.MULTI_THREAD_DECODE,
            threshold=cfg.DATA.THRESHOLD,
            use_sampling_ratio=cfg.DATA.USE_SAMPLING_RATIO,
            video_ids=video_ids,
            include_labels=include_labels,
            extend_classes=cfg.DATA.EXTEND_CLASSES,
        )

    # Here we do a cheeky swap with the wrapper and the dataset we actually want to return
    # so essentially we are supplying the dataloader our ConfounderWrapper and returning the
    # underlying dataset a.k.a return_train_dataset. The reason is that downstream code might use
    # properties on `Diving48Dataset` that is not available on `Diving48ConfounderDatasetWrapper` ;)
    if cfg.CONFOUNDERS.ENABLED:
        train_dataset = Diving48ConfounderDatasetWrapper(
            return_train_dataset,
            cfg
        )
    else:
        train_dataset = return_train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=True,
    )
    return train_loader, return_train_dataset


def get_train_objects(cfg: Config, model, device: torch.device = torch.device('cpu'), video_ids=None):
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    train_loader, train_dataset = get_train_loader_and_dataset(cfg, video_ids)

    if cfg.EGL.ENABLED:
        criterion = DualPathRRRLoss(lambdas=cfg.RRR.LAMBDAS, skip_zero_masks=True)
    else:
        if cfg.MODEL.CLASS_WEIGHTS:
            weights = torch.tensor(train_dataset.get_inverted_class_weights(), dtype=torch.float32)
            criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    scaler = None
    if cfg.TRAIN.AMP:
        scaler = GradScaler()

    return criterion, optimiser, train_loader, train_dataset, scaler


def forward(model, xb, device, cfg, scaler=None):
    xb_fast = xb[:].to(device)
    # reduce the number of frames by the alpha ratio
    # B x C x T / alpha x H x W
    xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)

    if scaler and torch.cuda.is_available():
        with autocast(device_type='cuda', dtype=torch.float16):
            return model([xb_slow, xb_fast])

    return model([xb_slow, xb_fast])


def backward(loss, scaler=None):
    if scaler:
        return scaler.scale(loss).backward()

    return loss.backward()
