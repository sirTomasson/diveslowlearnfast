import copy
import os.path

import torch

from tqdm import tqdm

from diveslowlearnfast.config import parse_args, save_config, to_dict
from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.models import SlowFast, save_checkpoint, load_checkpoint
from diveslowlearnfast.models.utils import last_checkpoint
from diveslowlearnfast.train import train_epoch, run_warmup

from pytorchvideo.transforms import ShortSideScale, Div255, Normalize
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from diveslowlearnfast.transforms import CenterCropVideo, Permute, ToTensor4D

def print_device_props(device):
    print(f'Running on {device}')

    if device == torch.device('cpu'): return

    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"Device: {device_props.name}")
    print(f"Total memory: {device_props.total_memory / 1024 ** 2:.2f} MB")
    print(f"GPU number: {device_props.major}.{device_props.minor}")


def main():
    cfg = parse_args()
    dict_cfg = to_dict(copy.deepcopy(cfg))
    print(dict_cfg)
    save_config(dict_cfg, os.path.join(cfg.TRAIN.RESULT_DIR, 'config.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_props(device)

    model = SlowFast(cfg).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    start_epoch = 1
    checkpoint_path = last_checkpoint(cfg.TRAIN.RESULT_DIR)
    if cfg.TRAIN.AUTO_RESUME and checkpoint_path is not None:
        model, optimiser, epoch = load_checkpoint(
            model,
            optimiser,
            checkpoint_path,
            device
        )
        start_epoch = epoch + 1

    transform = Compose([
        ToTensor4D(),
        Permute(3, 0, 1, 2),
        Div255(),  # Div255 assumes C x T x W x H
        ShortSideScale(size=256),
        # center crop
        CenterCropVideo(size=224),
        # UniformTemporalSubsample(32)
        # ensures each sample has the same number of frames, will under sample if T < 128, and over sample if T > 128
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    ])

    train_dataset = Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        dataset_type='train',
        transform_fn=transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD
    )

    dataloader = DataLoader(
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
        transform_fn=transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD
    )

    if checkpoint_path is None:
        run_warmup(model, optimiser, criterion, dataloader, device, cfg)

    epoch_bar = tqdm(range(start_epoch, cfg.SOLVER.MAX_EPOCH), desc=f'Train epoch')
    for epoch in epoch_bar:
        acc, loss = train_epoch(
            model,
            criterion,
            optimiser,
            dataloader,
            device,
            cfg
        )
        epoch_bar.set_postfix({
            'acc': f'{acc:.3f}',
            'loss': f'{loss:.3f}'
        })

        if epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            save_checkpoint(model,
                            optimiser,
                            epoch,
                            cfg.TRAIN.RESULT_DIR)

        # if epoch % cfg.TRAIN.EVAL_PERIOD == 0:

        print('\n')
        print(10 * '_')


if __name__ == '__main__':
    main()
