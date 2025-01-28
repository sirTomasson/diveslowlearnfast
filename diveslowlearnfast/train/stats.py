import unittest

import numpy as np
import pandas as pd


class PerSampleStatistics:
    def __init__(self, path=None):
        super().__init__()
        if path:
            self.df = pd.read_pickle(path)
        else:
            self.df = pd.DataFrame({'video_id': [], 'n_correct': [], 'total': [], 'class_id': []})

    def update(self, video_ids, y_pred, y_true):
        n_corrects = np.uint8(y_pred == y_true)
        class_ids = y_true
        for video_id, n_correct, class_id in zip(video_ids, n_corrects, class_ids):
            self._update_row(video_id, n_correct, class_id)

    def _update_row(self, video_id: str, n_correct: int, class_id: int):
        mask = self.df['video_id'] == video_id
        if mask.any():
            self.df.loc[mask, 'n_correct'] += n_correct
            self.df.loc[mask, 'total'] += 1
        else:
            self.df.loc[len(self.df)] = [video_id, n_correct, 1, class_id]

    def per_video_accuracy(self):
        self.df['acc'] = self.df['n_correct'] / self.df['total']
        return self.df

    def accuracy(self):
        return self.df['n_correct'].sum() / self.df['total'].sum()

    def per_class_accuracy(self):
        df = self.df.groupby(['class_id'])[['n_correct', 'total']].sum()
        df['per_class_acc'] = df['n_correct'].sum() / df['total'].sum()
        return df

    def save(self, path):
        self.df.to_pickle(path)


class PerSampleStatisticsTest(unittest.TestCase):

    def test_update(self):
        stats = PerSampleStatistics()
        video_ids = ['1', '2', '3', '4']
        y_pred = np.array([0, 1, 2, 2])
        y_true = np.array([0, 0, 2, 1])

        stats.update(video_ids, y_pred, y_true)
        self.assertEqual(stats.per_video_accuracy().loc[0, 'acc'], 1.0)
        self.assertEqual(stats.per_video_accuracy().loc[0, 'class_id'], 0)
        self.assertEqual(stats.per_video_accuracy().loc[1, 'acc'], 0.0)
        self.assertEqual(stats.per_video_accuracy().loc[1, 'class_id'], 0)
        self.assertEqual(stats.per_video_accuracy().loc[2, 'acc'], 1.0)
        self.assertEqual(stats.per_video_accuracy().loc[2, 'class_id'], 2)
        self.assertEqual(stats.per_video_accuracy().loc[3, 'acc'], 0.0)
        self.assertEqual(stats.per_video_accuracy().loc[3, 'class_id'], 1)

        self.assertEqual(stats.accuracy(), 0.5)

        y_pred = np.array([0, 1, 2, 2])
        y_true = np.array([0, 0, 1, 2])

        stats.update(video_ids, y_pred, y_true)
        self.assertEqual(stats.per_video_accuracy().loc[0, 'acc'], 1.0)
        self.assertEqual(stats.per_video_accuracy().loc[0, 'class_id'], 0)
        self.assertEqual(stats.per_video_accuracy().loc[1, 'acc'], 0.0)
        self.assertEqual(stats.per_video_accuracy().loc[1, 'class_id'], 0)
        self.assertEqual(stats.per_video_accuracy().loc[2, 'acc'], 0.5)
        self.assertEqual(stats.per_video_accuracy().loc[2, 'class_id'], 2)
        self.assertEqual(stats.per_video_accuracy().loc[3, 'acc'], 0.5)
        self.assertEqual(stats.per_video_accuracy().loc[3, 'class_id'], 1)

        self.assertEqual(stats.accuracy(), 0.5)

        per_class_acc_df = stats.per_class_accuracy()
        self.assertEqual(per_class_acc_df.loc[0, 'per_class_acc'], 0.5)
        self.assertEqual(per_class_acc_df.loc[1, 'per_class_acc'], 0.5)
        self.assertEqual(per_class_acc_df.loc[2, 'per_class_acc'], 0.5)


class Statistics:

    def __init__(self, **kwargs):
        self.stats = {
            'loss': [],
            'accuracy': [],
            'loader_time': [],
            'io_time': [],
            'batch_time': [],
            'transform_time': [],
        }
        for key, value in kwargs.items():
            self.stats[key] = value

        def _inner(key):
            return lambda: self.get_stat(key)

        for _key, value in self.stats.items():
            setattr(self, f'mean_{_key}', _inner(f'mean_{_key}'))
            setattr(self, f'current_{_key}', _inner(f'current_{_key}'))
            setattr(self, _key, _inner(_key))

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.stats[key].append(value)

    def get_formatted_stats(self, *args):
        result = {}
        for key in args:
            result[key] = self.get_stat(key, formatted=True)

        return result

    def get_stat(self, key, formatted=False):
        if key.startswith('current_'):
            value = self.stats[key.replace('current_', '')]
            value = value[-1] if len(value) > 0 else None
            if formatted:
                value = f'{value:.3f}' if value is not None else value

            return value
        elif key.startswith('mean_'):
            value = np.mean(self.stats[key.replace('mean_', '')])
            if formatted:
                value = f'{value:.3f}'

            return value
        else:
            return self.stats[key]


class StatisticsTest(unittest.TestCase):

    def test_update(self):
        stats = Statistics()

        stats.update(accuracy=0.25, loss=3.0)
        self.assertEqual(stats.accuracy(), [0.25])
        self.assertEqual(stats.loss(), [3.0])
        self.assertEqual(stats.mean_accuracy(), 0.25)
        self.assertEqual(stats.mean_loss(), 3.0)

        stats.update(accuracy=0.25, loss=3.0)
        self.assertEqual(stats.accuracy(), [0.25, 0.25])
        self.assertEqual(stats.loss(), [3.0, 3.0])
        self.assertEqual(stats.mean_accuracy(), 0.25)
        self.assertEqual(stats.mean_loss(), 3.0)

        stats.update(accuracy=0.5, loss=6.0)
        self.assertEqual(stats.accuracy(), [0.25, 0.25, 0.5])
        self.assertEqual(stats.loss(), [3.0, 3.0, 6.0])
        self.assertAlmostEqual(stats.mean_accuracy(), 0.33, places=2)
        self.assertEqual(stats.mean_loss(), 4.0)
