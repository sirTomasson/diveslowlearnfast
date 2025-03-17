import time
import os
import unittest
import sqlite3

import pandas as pd
import numpy as np

from diveslowlearnfast.config import Config

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from contextlib import contextmanager
from typing import List


def get_tuple(result: List[List], **_kwargs):
    return tuple(map(lambda x: x[0], result))


def get_value(result: List[List], **_kwargs):
    return get_tuple(result)[0]


def get_dict(result: List[List], columns: List[str], **_kwargs):
    return {column: get_column(idx)(result) for idx, column in enumerate(columns)}


def get_df(result: List[List], columns: List[str], **_kwargs):
    return pd.DataFrame(get_dict(result, columns=columns))


def get_column(column_idx):
    def _inner(result, **_kwargs):
        return list(map(lambda x: x[column_idx], result))

    return _inner


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
        cur.execute("""CREATE TABLE IF NOT EXISTS losses
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER,
            created_at INTEGER,
            tag INTEGER,
            value REAL,
            run_id TEXT, 
            split TEXT
        )
        """)
        self.con.commit()

    def add_loss(self, loss, epoch, tag, run_id, split):
        query = """INSERT INTO losses(epoch, created_at, tag, value, run_id, split) 
        VALUES(?, ?, ?, ?, ?, ?)
        """
        data = [epoch, int(time.time() * 1000.0), tag, loss, run_id, split]
        cur = self.con.cursor()
        cur.execute(query, data)
        self.con.commit()

    def update(self, video_ids, preds, gts, run_id, mode, epoch):
        query = """INSERT INTO stats(epoch, video_id, pred, gt, run_id, split) 
        VALUES(?, ?, ?, ?, ?, ?)
        """
        # explicitly cast to ints since the numpy definition is not supported by the db
        data = [[epoch, video_id, int(pred), int(gt), run_id, mode] for video_id, pred, gt in
                zip(video_ids, preds, gts)]
        cur = self.con.cursor()
        cur.executemany(query, data)
        self.con.commit()

    def execute_query(self, query, *args, data=None, extractor=None, return_columns=False, **kwargs):
        if data is None and args is not None:
            data = tuple(args)

        cursor = self.con.cursor()
        try:
            if data is None:
                result = [list(row) for row in cursor.execute(query)]
            else:
                result = [list(row) for row in cursor.execute(query, data)]

            columns = [description[0] for description in cursor.description]

            if extractor is None:
                if return_columns:
                    return columns, result
                return result

            result = extractor(result, columns=columns)
            if return_columns:
                return columns, result

            return result
        finally:
            cursor.close()

        return None


    def get_sample_accuracy(self, epoch_start, run_id, split, epoch_end: int | str = 'max_epoch', **kwargs):
        data = [epoch_start]
        if epoch_end == 'max_epoch':
            epoch_end = ''
        else:
            data.append(epoch_end)
            epoch_end = 'AND epoch <= ?'

        data.extend([run_id, split])
        query = f"""SELECT video_id, gt, (correct_n / n) as acc FROM(
            SELECT
                video_id,
                gt,
                epoch,
                CAST(SUM(CASE WHEN pred = gt THEN 1 ELSE 0 END) as REAL) as correct_n,
                CAST(COUNT(*) as REAL) as n
            FROM stats
            WHERE epoch > ? {epoch_end}
            AND run_id = ?
            AND split = ?
            GROUP BY video_id, gt
        ) ORDER BY acc
        """
        return self.execute_query(query, *data, **kwargs)

    def get_lowest_percentile(self, epoch_start, run_id, split, epoch_end: int | str = 'max_epoch', percentile=5,
                              **kwargs):
        sample_acc = self.get_sample_accuracy(epoch_start, run_id, split, epoch_end, **kwargs)
        top_n = int(len(sample_acc) * (percentile / 100))
        return sample_acc[:top_n]

    def get_below_median_samples(self, epoch_start, run_id, split, epoch_end: int | str = 'max_epoch', **kwargs):
        data = [epoch_start]
        if epoch_end == 'max_epoch':
            epoch_end = ''
        else:
            data.append(epoch_end)
            epoch_end = 'AND epoch <= ?'

        data.extend([run_id, split])
        query = f"""
        WITH
            acc AS (
                SELECT video_id, gt, (correct_n / n) as acc FROM(
                    SELECT
                        video_id,
                        gt,
                        epoch,
                        CAST(SUM(CASE WHEN pred = gt THEN 1 ELSE 0 END) as REAL) as correct_n,
                        CAST(COUNT(*) as REAL) as n
                    FROM stats
                    WHERE epoch > ? {epoch_end}
                    AND run_id = ?
                    AND split = ?
                    GROUP BY video_id, gt
                )
            ),
            median AS (
                SELECT AVG(acc) as median FROM(
                    SELECT * FROM acc
                    ORDER BY acc
                    LIMIT 2 - (SELECT COUNT(*) FROM acc) % 2
                    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM acc)
                )
            )
        SELECT *, (SELECT median FROM median) as median FROM acc
        WHERE acc < median
        ORDER BY acc
        """
        return self.execute_query(query, *data, **kwargs)

    def get_ytrue_and_pred(self, epoch, run_id, split, **kwargs):
        result = self.execute_query("""SELECT gt, pred FROM stats
        WHERE epoch = ? AND split = ? AND run_id = ?
        """, epoch, split, run_id, **kwargs)
        result = np.array(result)
        Y_true = result[:, 0]
        Y_pred = result[:, 1]
        labels = np.unique(Y_true)
        return Y_true, Y_pred, labels

    def confusion_matrix(self, epoch, run_id, split, **kwargs):
        Y_true, Y_pred, labels = self.get_ytrue_and_pred(epoch, run_id, split, **kwargs)
        return confusion_matrix(Y_true, Y_pred, labels=labels), labels


    def per_class_accuracy(self, epoch, run_id, split, **kwargs):
        matrix, labels = self.confusion_matrix(epoch, run_id, split, **kwargs)
        totals = np.array(matrix.sum(axis=1)) + 1e-9
        diagonals = np.array(matrix.diagonal())
        return (diagonals / totals), labels



def wps_strategy(stats_db: StatsDB, cfg: Config):
    strategy = cfg.EGL.WORST_PERFORMER_STRATEGY
    assert strategy in ['median', 'percentile']
    if strategy == 'median':
        return stats_db.get_below_median_samples
    elif strategy == 'percentile':
        def _wsp_percentile_strategy(epoch_start, run_id, split, epoch_end: int | str = 'max_epoch'):
            return stats_db.get_lowest_percentile(epoch_start, run_id, split, epoch_end,
                                                  percentile=cfg.EGL.WORST_PERFORMER_PERCENTILE)

        return _wsp_percentile_strategy


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
            stat = self.stats[key.replace('mean_', '')]
            if len(stat) == 0:
                return None

            value = np.mean(stat)
            if formatted:
                value = f'{value:.3f}'

            return value
        else:
            return self.stats[key]

    @contextmanager
    def timer(self, key):
        start = time.time()
        yield
        elapsed = time.time() - start
        self.update(**{key: elapsed})


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

    def test_timer(self):
        stats = Statistics()
        with stats.timer('batch_time'):
            _ = [i * i for i in range(1000)]

        self.assertGreater(stats.mean_batch_time(), 0.0)
        self.assertIsNone(stats.mean_loss())
