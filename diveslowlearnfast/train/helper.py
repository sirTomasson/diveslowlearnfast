from typing import Iterator

import pytorchvideo
import torch

import torch.nn as nn

from pytorchvideo.transforms import Div255, Normalize
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets import Diving48Dataset, Diving48ConfounderDatasetWrapper
from diveslowlearnfast.loss.dice import DiceLoss
from diveslowlearnfast.loss.rrr import RRRLoss
from diveslowlearnfast.train.stats import Statistics
from diveslowlearnfast.transforms import ToTensor4D, Permute, RandomRotateVideo, KwargsCompose, CutoutSegment, \
    RandomApply, DeterministicRandomShortSideScale, DeterministicRandomCrop, DeterministicHorizontalFlip


def get_batch(loader: Iterator,
              device: torch.device,
              stats: Statistics = None,
              data_requires_grad: bool = False):
    if stats:
        with stats.timer('loader_time'):
            xb, yb, io_times, transform_times, video_ids, masks = next(loader)
    else:
        xb, yb, io_times, transform_times, video_ids, masks = next(loader)

    if stats:
        stats.update(
            io_time=io_times.numpy().mean(),
            transform_time=transform_times.numpy().mean()
        )
    xb = xb.to(device)
    xb.requires_grad = data_requires_grad
    return xb, yb.to(device), video_ids, masks


def get_randaug_transform(cfg: Config, crop_size, p=.5):
    return KwargsCompose([
        ToTensor4D(dtype=torch.uint8),
        Div255(),
        DeterministicRandomShortSideScale(),
        DeterministicRandomCrop(crop_size),
        Permute(1, 0, 2, 3),
        pytorchvideo.transforms.rand_augment.RandAugment(
            num_layers=cfg.RAND_AUGMENT.NUM_LAYERS,
            magnitude=cfg.RAND_AUGMENT.MAGNITUDE,
            prob=p,
        ),
        Permute(1, 0, 2, 3),
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        DeterministicHorizontalFlip(p=0.5),
    ])


def get_rotate_transform(cfg: Config, crop_size):
    angle = cfg.RANDOM_ROTATE.MAX_DEGREE
    return KwargsCompose([
        ToTensor4D(),
        Div255(),
        DeterministicRandomShortSideScale(),
        DeterministicRandomCrop((crop_size, crop_size)),
        Permute(1, 0, 2, 3),  # From 3 x T x H X W -> T x 3 x H x W
        RandomRotateVideo(-angle, angle),
        Permute(1, 0, 2, 3),  # From T x 3 x H X W -> 3 x T x H x W
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        DeterministicHorizontalFlip(p=0.5),
    ])


def get_cutout_segment_transform(cfg: Config, crop_size, p=.5):
    return KwargsCompose([
        ToTensor4D(),
        Permute(1, 2, 3, 0),  # From 3 x T x H x W -> T x H X W x 3
        CutoutSegment(dataset_path=cfg.CUTOUT_SEGMENT.SEGMENTS_PATH, p=p),
        Permute(3, 0, 1, 2),  # From T x H X W x 3 -> 3 x T x H x W
        Div255(),
        DeterministicRandomShortSideScale(
            min_size=256,
            max_size=320,
        ),
        DeterministicRandomCrop(crop_size),
        DeterministicHorizontalFlip(p=0.5),
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    ])


def get_base_transform(cfg, crop_size):
    return KwargsCompose([
        ToTensor4D(),
        Div255(),
        DeterministicRandomShortSideScale(
            min_size=256,
            max_size=320,
        ),
        DeterministicRandomCrop((crop_size, crop_size)),
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        DeterministicHorizontalFlip(p=0.5),
    ])


def get_train_transform(cfg: Config, crop_size=None):
    crop_size = cfg.DATA.TRAIN_CROP_SIZE if crop_size is None else crop_size
    if cfg.RANDOM_APPLY_TRANSFORM.ENABLED:
        transformations = list(filter(lambda x: x is not None, [
            get_randaug_transform(cfg, crop_size, p=1.0) if cfg.RAND_AUGMENT.ENABLED else None,
            get_rotate_transform(cfg, crop_size) if cfg.RANDOM_ROTATE.ENABLED else None,
            get_cutout_segment_transform(cfg, crop_size, p=1.0) if cfg.CUTOUT_SEGMENT.ENABLED else None,
            get_base_transform(cfg, crop_size)
        ]))
        assert len(transformations) > 0, 'At least one transform must be enabled'
        return RandomApply(transformations, p=cfg.RANDOM_APPLY_TRANSFORM.PROB)
    elif cfg.RAND_AUGMENT.ENABLED:
        return get_randaug_transform(cfg, crop_size, p=cfg.RAND_AUGMENT.PROB)
    elif cfg.RANDOM_ROTATE.ENABLED:
        return get_rotate_transform(cfg, crop_size)
    elif cfg.CUTOUT_SEGMENT.ENABLED:
        return get_cutout_segment_transform(cfg, crop_size)
    else:
        return get_base_transform(cfg, crop_size)


def get_test_transform(cfg: Config):
    return KwargsCompose([
        ToTensor4D(),
        Div255(),
        pytorchvideo.transforms.create_video_transform(
            mode='test',
            num_samples=cfg.DATA.NUM_FRAMES,
            video_std=cfg.DATA.STD,
            video_mean=cfg.DATA.MEAN,
            convert_to_float=False,
            crop_size=cfg.DATA.TEST_CROP_SIZE,
        ),
    ])


def _binary_mask_to_float_tensor(mask):
    T, H, W = mask.shape
    mask = torch.from_numpy(mask).float().unsqueeze(1)
    mask = mask.expand(T, 3, H, W)
    return mask.permute(1, 0, 2, 3)


def _float_tensor_to_binary_mask(mask):
    mask = mask.permute(1, 0, 2, 3)
    return mask[:, 0, :, :].to(dtype=torch.bool).unsqueeze(0)


def get_mask_transform(cfg: Config):
    return KwargsCompose([
        _binary_mask_to_float_tensor,
        DeterministicRandomShortSideScale(min_size=256, max_size=320),
        DeterministicRandomCrop((cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE)),
        DeterministicHorizontalFlip(),
        _float_tensor_to_binary_mask
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
        prefetch_factor=4
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


def get_mask_type(cfg: Config):
    if cfg.EGL.METHOD == 'confounder':
        return 'confounder'
    elif cfg.EGL.METHOD == 'cache':
        return 'cache'
    elif cfg.EGL.METHOD == 'ogl':
        return 'segments'
    else:
        return None


def get_train_loader_and_dataset(cfg, video_ids=None):
    train_transform = get_train_transform(cfg)

    include_labels = None
    if len(cfg.DATA.INCLUDE_LABELS) > 0:
        include_labels = cfg.DATA.INCLUDE_LABELS

    if cfg.EGL.ENABLED:
        mask_type = get_mask_type(cfg)
        return_train_dataset = Diving48Dataset(
            cfg.DATA.DATASET_PATH,
            cfg.DATA.NUM_FRAMES,
            alpha=cfg.SLOWFAST.ALPHA,
            dataset_type='train',
            transform_fn=train_transform,
            mask_transform_fn=get_mask_transform(cfg),
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
            mask_type=mask_type,
            loader_mode=cfg.DATA.FORMAT,
            crop_size=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
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
            loader_mode=cfg.DATA.FORMAT,
            crop_size=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
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
        prefetch_factor=4
    )
    return train_loader, return_train_dataset


def get_criterion(cfg: Config, dataset: Diving48Dataset, device: torch.device):
    if cfg.MODEL.CLASS_WEIGHTS:
        weights = torch.tensor(dataset.get_inverted_class_weights(), dtype=torch.float32)
        ce_loss = nn.CrossEntropyLoss(weight=weights).to(device)
    else:
        ce_loss = nn.CrossEntropyLoss()

    if cfg.EGL.ENABLED:
        assert cfg.EGL.LOSS_FUNC in ['rrr', 'rrr_v2', 'dice']
        if cfg.EGL.LOSS_FUNC in ['rrr', 'rrr_v2']:
            criterion = RRRLoss(cfg.RRR.LAMBDAS, skip_zero_masks=True)
        else:
            criterion = DiceLoss(ce_loss, cfg.DICE.SMOOTH, cfg.DICE.ALPHA, cfg.DICE.BETA)
    else:
        criterion = ce_loss

    return criterion


def get_train_objects(cfg: Config, model, device: torch.device = torch.device('cpu'), video_ids=None):
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    train_loader, train_dataset = get_train_loader_and_dataset(cfg, video_ids)

    criterion = get_criterion(cfg, train_dataset, device)

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
