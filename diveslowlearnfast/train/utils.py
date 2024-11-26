import json
import os


def save_stats(stats: dict, save_dir):
    stats_path = os.path.join(save_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f)


def load_stats(stats_path) -> dict:
    if not os.path.exists(stats_path):
        return {
            'train_losses': [],
            'train_accuracies': [],
            'test_losses': [],
            'test_accuracies': []
        }
    with open(stats_path, 'r') as f:
        return json.load(f)
