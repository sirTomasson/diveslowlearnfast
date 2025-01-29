import os
import unittest

import numpy as np
import sqlite3

class StatsDB:
    def __init__(self, path=None):
        super().__init__()
        self.con = sqlite3.connect(path)
        cur = self.con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS stats 
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            epoch INTEGER, 
            video_id TEXT, 
            pred INTEGER, 
            gt INTEGER, 
            run_id TEXT, 
            split TEXT
        )
        """)
        self.con.commit()

    def update(self, video_ids, preds, gts, run_id, mode, epoch):
        query = """INSERT INTO stats(epoch, video_id, pred, gt, run_id, split) 
        VALUES(?, ?, ?, ?, ?, ?)
        """
        # explicitly cast to ints since the numpy definition is not supported by the db
        data = [[epoch, video_id, int(pred), int(gt), run_id, mode] for video_id, pred, gt  in zip(video_ids, preds, gts)]
        cur = self.con.cursor()
        cur.executemany(query, data)
        self.con.commit()


    def execute_query(self, query, **kwargs):
        return [list(row) for row in self.con.execute(query, kwargs)]



class PerSampleStatisticsTest(unittest.TestCase):

    def test_update(self):
        stats = StatsDB('./stats.db')
        video_ids = ['1', '2', '3']
        y_pred = [0, 5, 3]
        y_true = [0, 5, 4]
        epoch = 1
        run_id = 'run1'
        mode = 'train'

        stats.update(video_ids, y_pred, y_true, run_id, mode, epoch)
        query = """SELECT gt, (correct_n / n) as acc FROM(
        SELECT
            gt,
            epoch,
            CAST(SUM(CASE WHEN pred = gt THEN 1 ELSE 0 END) as REAL) as correct_n,
            CAST(COUNT(*) as REAL) as n
        FROM stats
        GROUP BY gt);
        """
        per_class_accuracy = stats.execute_query(query)
        self.assertEqual(per_class_accuracy, [[0, 1.0], [4, 0.0], [5, 1.0]])

        video_ids = ['2', '1', '3']
        y_pred = [1, 0, 1]
        y_true = [0, 0, 2]
        epoch = 2
        run_id = 'run1'
        mode = 'train'
        stats.update(video_ids, y_pred, y_true, run_id, mode, epoch)
        per_class_accuracy = stats.execute_query(query)
        self.assertEqual(per_class_accuracy, [[0, 0.6666666666666666], [2, 0.0], [4, 0.0], [5, 1.0]])

    def tearDown(self):
        if os.path.exists('./stats.db'):
            os.remove('./stats.db')



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
