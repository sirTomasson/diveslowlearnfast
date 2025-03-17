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


def plot_confusion_matrix(confusion_matrix, save_path=None, labels=None, **_kwargs):
    if labels is None:
        labels = np.arange(len(confusion_matrix))

    # Visualizing the confusion matrix using matplotlib
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(confusion_matrix)), labels=labels)
    plt.yticks(np.arange(len(confusion_matrix)), labels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Adding text annotations for each cell
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            plt.text(j, i, format(confusion_matrix[i][j], 'd'),
                     ha="center", va="center", color="black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_per_class_accuracy(confusion_matrix, stats_path=None, labels=None):
    # to avoid divide by zero
    totals = np.array(confusion_matrix.sum(axis=1)) + 1e-9
    diagonals = np.array(confusion_matrix.diagonal())
    per_class_accuracy = (diagonals / totals) * 100

    if labels is None:
        labels = range(len(per_class_accuracy))

    # Create continuous indices and map them to actual labels
    continuous_indices = range(len(labels))

    plt.figure(figsize=(15, 10))

    # Create main axes
    ax1 = plt.gca()
    ax1.bar(continuous_indices, per_class_accuracy, width=0.8, color='C0')
    ax1.set_xticks(continuous_indices)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Class ID')

    # Create top x-axis with same transform as bottom axis
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(continuous_indices)
    ax2.set_xticklabels([f'{val:.0f}' for val in per_class_accuracy], rotation=45, ha='right')
    ax2.set_xlabel('Accuracy (%)')

    # Create right y-axis for sample counts
    ax_right = ax1.twinx()
    ax_right.set_ylabel('Number of Samples')
    ax_right.bar(continuous_indices, totals, width=0.8, color='lightgray', alpha=0.5)

    # Create custom legend
    import matplotlib.patches as mpatches
    accuracy_patch = mpatches.Patch(color='C0', label='Accuracy')
    samples_patch = mpatches.Patch(color='lightgray', alpha=0.5, label='Number of Samples')

    # Add legend
    ax1.legend(handles=[accuracy_patch, samples_patch],
               loc='upper right',
               bbox_to_anchor=(1, 1),
               frameon=True,
               framealpha=1.0,
               edgecolor='black')

    plt.grid(axis='y')
    plt.tight_layout()
    if stats_path is not None:
        plt.savefig(stats_path.parent / 'per_class_accuracy.png')

    plt.show()


def eval_stats(stats_path, **_kwargs):
    with open(stats_path, 'rb') as f:
        stats = json.load(f)['eval']

    save_path = stats_path.parent / 'confusion_matrix.png'
    confusion_matrix = np.array(stats['confusion_matrix'])
    labels = stats['labels']
    plot_confusion_matrix(confusion_matrix, save_path, labels)
    plot_per_class_accuracy(confusion_matrix, save_path, labels)

    acc, precision, recall, f1 = stats['acc'], stats['precision'], stats['recall'], stats['f1']

    # Plotting accuracy, precision, recall, and f1 in a 2x2 matrix
    metrics = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

    bars = plt.bar(metrics.keys(), metrics.values(), label=metrics)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.title('Metrics')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.xlabel('Metric')
    plt.ylabel('Score (0-1)')
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
