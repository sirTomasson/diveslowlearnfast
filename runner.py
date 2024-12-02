import copy
import os.path

import pytorchvideo.transforms
import torch

from tqdm import tqdm

from diveslowlearnfast.config import parse_args, save_config, to_dict
from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.models import SlowFast, save_checkpoint, load_checkpoint, get_parameter_count
from diveslowlearnfast.models.utils import last_checkpoint
from diveslowlearnfast.train import run_train_epoch, run_warmup, save_stats, load_stats, run_test_epoch

from pytorchvideo.transforms import Div255, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from diveslowlearnfast.transforms import Permute, ToTensor4D

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

    train_transform = Compose([
        ToTensor4D(),
        Permute(3, 0, 1, 2), # From T x H X W x 3 -> 3 x T x H x W
        Div255(),
        pytorchvideo.transforms.create_video_transform(
            mode='train',
            num_samples=cfg.DATA.NUM_FRAMES,
            video_std=cfg.DATA.MEAN,
            video_mean=cfg.DATA.STD,
            convert_to_float=False,
            crop_size=224,
            horizontal_flip_prob=0.5,
            random_resized_crop_paras={'scale': (1.0, 1.0), 'aspect_ratio': (1.0, 1.0)}
        ),
        Permute(1, 0, 2, 3), # From 3 x T x H X W -> T x 3 x H x W
        RandAugment(prob=0.5, sampling_type='gaussian'),
        Permute(1, 0, 2, 3), # From T x 3 x H X W -> 3 x T x H x W
    ])

    test_transform = Compose([
        ToTensor4D(),
        Permute(3, 0, 1, 2),
        Div255(),
        pytorchvideo.transforms.create_video_transform(
            mode='test',
            num_samples=cfg.DATA.NUM_FRAMES,
            video_std=cfg.DATA.MEAN,
            video_mean=cfg.DATA.STD,
            convert_to_float=False,
            crop_size=224,
        ),
    ])

    train_dataset = Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        dataset_type='train',
        transform_fn=train_transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD,
        temporal_random_jitter=cfg.DATA.TEMPORAL_RANDOM_JITTER,
        temporal_random_offset=cfg.DATA.TEMPORAL_RANDOM_OFFSET
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
        use_decord=cfg.DATA_LOADER.USE_DECORD
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=False,
    )

    print(f'Start training model:')
    parameter_count = get_parameter_count(model)
    print(f'parameter count = {parameter_count}')
    # 4bytes x 3 (1model + 1gradient + 1optimiser)
    total_parameter_bytes = parameter_count * 12
    print(f'model size      = {total_parameter_bytes / 1024 ** 2:.3f} MB')
    print(f'from checkpoint = {checkpoint_path}')

    if checkpoint_path is None:
        run_warmup(model, optimiser, criterion, train_loader, device, cfg)
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    model.train()
    stats = load_stats(os.path.join(cfg.TRAIN.RESULT_DIR, 'stats.json'))
    epoch_bar = tqdm(range(start_epoch, cfg.SOLVER.MAX_EPOCH), desc=f'Train epoch')
    for epoch in epoch_bar:
        train_acc, train_loss = run_train_epoch(
            model,
            criterion,
            optimiser,
            train_loader,
            device,
            cfg,
        )
        epoch_bar.set_postfix({
            'acc': f'{train_acc:.3f}',
            'train_loss': f'{train_loss:.3f}'
        })

        if epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            save_checkpoint(model,
                            optimiser,
                            epoch,
                            cfg.TRAIN.RESULT_DIR)

        stats['train_losses'].append(train_loss)
        stats['train_accuracies'].append(train_acc)

        if epoch % cfg.TRAIN.EVAL_PERIOD == 0:
            model.eval()
            test_acc, test_loss = run_test_epoch(
                model,
                criterion,
                test_loader,
                device,
                cfg,
            )
            model.train()
            stats['test_losses'].append(test_acc)
            stats['test_accuracies'].append(test_loss)

        save_stats(stats, cfg.TRAIN.RESULT_DIR)

        print('\n')
        print(10 * '_')


if __name__ == '__main__':
    main()
