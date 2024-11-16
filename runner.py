import torch

import numpy as np

from datetime import datetime

from diveslowlearnfast.config import (
    merge_config,
    Config,
    parse_args
)

from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.models import SlowFast
from diveslowlearnfast.train import train_epoch

from pytorchvideo.transforms import ShortSideScale, Div255, UniformTemporalSubsample
from pytorchvideo.transforms.functional import uniform_crop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms.v2 import Lambda


def main():
    cfg = Config()
    args = parse_args()
    cfg = merge_config(cfg, args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SlowFast(cfg).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    transform = Compose([
        Lambda(lambda x: torch.tensor(x).permute(3, 0, 1, 2)),
        Div255(),  # Div255 assumes C x T x W x H
        ShortSideScale(size=256),
        # center crop
        Lambda(lambda x: uniform_crop(x, size=224, spatial_idx=1)),
        UniformTemporalSubsample(32)
        # ensures each sample has the same number of frames, will under sample if T < 128, and over sample if T > 128
    ])

    dataset = Diving48Dataset(
        cfg.DATA.VIDEOS_PATH,
        cfg.DATA.ANNOTATIONS_PATH,
        transform_fn=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.TRAIN.PIN_MEMORY,
        shuffle=True,
    )

    epoch_times = []
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        start = datetime.now()
        train_epoch(
            model,
            criterion,
            optimiser,
            dataloader,
            device,
            cfg
        )
        epoch_times.append((datetime.now() - start).seconds)
        mean_time = np.mean(epoch_times)
        print(f'Epoch {epoch + 1}/{cfg.SOLVER.MAX_EPOCH} - took {epoch_times[-1]:.3f} s, average time {mean_time:.3f} s')


if __name__ == '__main__':
    main()
