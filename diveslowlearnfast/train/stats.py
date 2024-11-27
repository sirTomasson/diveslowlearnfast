import unittest

import numpy as np


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
        result = { }
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