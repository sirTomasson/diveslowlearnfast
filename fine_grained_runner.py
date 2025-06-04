import copy
import os
import torch

import torch.nn as nn
import numpy as np

from diveslowlearnfast import load_checkpoint, to_slowfast_inputs
from diveslowlearnfast.config import to_dict, save_config, parse_args, load_config, Config
from diveslowlearnfast.models import SlowFast
from diveslowlearnfast.models.utils import last_checkpoint, save_checkpoint
from runner import print_device_props
from diveslowlearnfast.train import helper as train_helper, load_stats, save_stats

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

@torch.no_grad()
def calculate_accuracy(logits, yb):
    B, T = yb.size()
    logits = logits.view(B, T, -1)
    pred = torch.argmax(logits, dim=-1)

    correct = (pred == yb).float()
    return correct.mean().item()


@torch.no_grad()
def run_test_epoch(model, loader, device, cfg: Config):
    loader_iter = iter(loader)
    batch_bar = tqdm(range(1, len(loader) + 1), desc=f'Test batch')

    accuracies = []
    for i in batch_bar:
        xb, yb, *_ = next(loader_iter)
        xb = xb.to(device)
        yb = yb.to(device)

        inputs = to_slowfast_inputs(xb, cfg.SLOWFAST.ALPHA)
        logits = model(inputs)
        accuracies.append(calculate_accuracy(logits, yb))

        batch_bar.set_postfix({ 'accuracy': np.mean(accuracies) })

    return np.mean(accuracies)

def run_train_epoch(model, loader, optimiser, criterion, device, cfg: Config):
    n_batches_per_step = cfg.TRAIN.MACRO_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
    n_batches_per_step = n_batches_per_step if n_batches_per_step > 0 else 1

    loader_iter = iter(loader)
    batch_bar = tqdm(range(1, len(loader) + 1), desc=f'Train batch')

    running_loss = 0.0
    accuracies = []
    losses = []
    for i in batch_bar:
        xb, yb, *_ = next(loader_iter)
        xb = xb.to(device)
        yb = yb.to(device)

        inputs = to_slowfast_inputs(xb, cfg.SLOWFAST.ALPHA)
        logits = model(inputs)
        loss = criterion(logits.view(-1, cfg.MODEL.NUM_CLASSES), yb.view(-1)) / n_batches_per_step
        loss.backward()

        running_loss += loss.item()
        accuracies.append(calculate_accuracy(logits, yb))

        if i % n_batches_per_step == 0 or i == len(loader) + 1:
            optimiser.step()
            optimiser.zero_grad()
            losses.append(running_loss)
            running_loss = 0.0
            print(len(losses))
            print(len(accuracies))
            batch_bar.set_postfix({ 'loss': np.mean(losses), 'accuracy': np.mean(accuracies) })

    return np.mean(accuracies), np.mean(losses)

def main():
    args = parse_args()
    checkpoint_filename = args.TRAIN.CHECKPOINT_FILENAME
    config_path = os.path.join(args.TRAIN.RESULT_DIR, 'config.json')

    if os.path.exists(config_path):
        logger.info(f'Loading config from {config_path}, arguments are ignored')
        cfg = load_config(config_path)
        cfg.DATA.DATASET_PATH = args.DATA.DATASET_PATH
    else:
        logger.info(f'Saving config to {config_path}')
        cfg = args
        dict_cfg = to_dict(copy.deepcopy(cfg))
        save_config(dict_cfg, config_path)

    # always override these settings, so we can reuse the config that was saved on disk but have different
    # behaviour depending on whether one of these was enabled
    if checkpoint_filename and len(checkpoint_filename) > 0:
        cfg.TRAIN.CHECKPOINT_FILENAME = checkpoint_filename

    dict_cfg = to_dict(copy.deepcopy(cfg))
    print(dict_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_props(device)

    model = SlowFast(cfg).to(device)
    criterion, optimiser, train_loader, train_dataset, scaler = train_helper.get_train_objects(cfg, model, device)
    test_loader = train_helper.get_test_objects(cfg)

    # If there is a checkpoint_path it is not worth using these weights
    if len(cfg.TRAIN.WEIGHTS_PATH) > 0:
        logger.info(f'Using pre-trained weights from {cfg.TRAIN.WEIGHTS_PATH}')
        load_checkpoint(model, cfg.TRAIN.WEIGHTS_PATH, device=device)

    if cfg.FINE_DIVING.ENABLED:
        projection = nn.Sequential(
            model.head.projection,
            nn.Linear(cfg.MODEL.NUM_CLASSES, cfg.DATA.NUM_FRAMES * cfg.MODEL.NUM_CLASSES, bias=True),
        )
        model.head.projection = projection

    start_epoch = 1
    checkpoint_path = last_checkpoint(cfg)
    if cfg.TRAIN.AUTO_RESUME and checkpoint_path is not None:
        model, optimiser, epoch = load_checkpoint(
            model,
            checkpoint_path,
            optimiser,
            device
        )
        start_epoch = epoch + 1

    stats = load_stats(os.path.join(cfg.TRAIN.RESULT_DIR, 'stats.json'))
    epoch_bar = tqdm(range(start_epoch, cfg.SOLVER.MAX_EPOCH), desc=f'Train epoch')
    for epoch in epoch_bar:

        accuracy, loss = run_train_epoch(model, train_loader, optimiser, criterion, device, cfg)
        stats['train_losses'].append(float(accuracy))
        stats['train_accuracies'].append(float(loss))

        if epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            save_checkpoint(model, optimiser, epoch, cfg)

        if epoch % cfg.TRAIN.EVAL_PERIOD == 0:
            model.eval()
            test_accuracy = run_test_epoch(model, test_loader, device, cfg)
            if len(stats['test_accuracies']) == 0:
                save_checkpoint(model, optimiser, epoch, cfg, 'best.pth')
            elif test_accuracy > stats['test_accuracies'][-1]:  # the current model is better than the previous model
                save_checkpoint(model, optimiser, epoch, cfg, 'best.pth')

            stats['test_accuracies'].append(float(test_accuracy))
            model.train()

        save_stats(stats, cfg.TRAIN.RESULT_DIR)

        print('\n')
        print(10 * '_')



if __name__ == '__main__':
    main()