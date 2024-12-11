import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

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
    fig, axis = plt.subplots(2, 2, figsize=(10, 8))
    with open(stats_path, 'rb') as f:
        stats = json.load(f)

    for ax, (k, v) in zip(axis.flatten(), stats.items()):
        ax.plot(v)
        ax.set_title(k)
        ax.grid()

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