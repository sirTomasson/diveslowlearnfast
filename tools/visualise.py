import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def display(stats_path, which='losses', mode='train'):
    assert which in ['accuracies', 'losses']
    assert mode in ['test', 'train']
    with open(stats_path, 'rb') as f:
        stats = json.load(f)

    stat = stats[f'{mode}_{which}']
    plt.plot(stat)
    plt.title(f'{mode}_{which}')
    plt.xlabel('iter')
    plt.ylabel(which)
    plt.grid()
    plt.show()


def display_all(stats_path, **kwargs):
    with open(stats_path, 'rb') as f:
        stats = json.load(f)


    test_losses = stats['test_losses']
    train_losses = stats['train_losses']

    if len(test_losses) > 0:
        x_old = np.arange(len(test_losses))
        x_new = np.linspace(0, len(test_losses), len(train_losses))
        test_losses = np.interp(x_new, x_old, test_losses)

        plt.plot(test_losses, label='test')

    plt.plot(train_losses, label='train')

    plt.title(f'{stats_path}: Loss')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    test_accuracies = stats['test_accuracies']
    train_accuracies = stats['train_accuracies']

    if len(test_accuracies) > 0:
        x_old = np.arange(len(test_accuracies))
        x_new = np.linspace(0, len(test_accuracies), len(train_losses))
        test_accuracies = np.interp(x_new, x_old, test_accuracies)

        plt.plot(test_accuracies, label='test')

    plt.plot(train_accuracies, label='train')

    plt.title(f'{stats_path}: Accuracy')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SlowFast network runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'stats_path',
        type=Path,
        help='Path to the stats file'
    )
    parser.add_argument(
        '--which', '-W',
        help='Which stat to display options are: losses, accuracies',
        default='loss'
    )
    parser.add_argument(
        '--mode', '-M',
        help='Mode of stat to display options are: train, test',
        default='train'
    )
    parser.add_argument(
        '--all', '-A',
        type=bool,
        default=False,
        action='store'
    )

    args = vars(parser.parse_args())
    if args['all']:
        display_all(**args)
    else:
        display(**args)