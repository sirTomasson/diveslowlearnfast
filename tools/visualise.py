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


def plot_confusion_matrix(confusion_matrix, save_path=None, **_kwargs):
    # Visualizing the confusion matrix using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(confusion_matrix)), labels=np.arange(len(confusion_matrix)))
    plt.yticks(np.arange(len(confusion_matrix)), labels=np.arange(len(confusion_matrix)))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Adding text annotations for each cell
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            plt.text(j, i, format(confusion_matrix[i][j], 'd'),
                     ha="center", va="center", color="black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def eval_stats(stats_path, **_kwargs):
    with open(stats_path, 'rb') as f:
        stats = json.load(f)['eval']

    save_path = stats_path.parent / 'confusion_matrix.png'
    plot_confusion_matrix(stats['confusion_matrix'], save_path)
    acc, precision, recall, f1 = stats['acc'], stats['precision'], stats['recall'], stats['f1']

    # Plotting accuracy, precision, recall, and f1 in a 2x2 matrix
    metrics = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Evaluation Metrics")

    for ax, (metric_name, value) in zip(axes.ravel(), metrics.items()):
        ax.bar([metric_name], [value], color='blue')
        ax.set_ylim(0, 1)
        ax.set_title(metric_name)
        ax.grid(axis='y')
        ax.set_ylabel('Score')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(stats_path.parent / 'eval_metrics.png')
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
    if args['mode'] == 'eval':
        eval_stats(**args)
    elif args['all']:
        display_all(**args)
    else:
        display(**args)
