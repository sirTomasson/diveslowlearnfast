import os
import json
import unittest
from argparse import Namespace
from dataclasses import is_dataclass
from pathlib import Path

from diveslowlearnfast.config import Config


def to_dict(value):
    cfg_dict = value.__dict__
    for k, v in cfg_dict.items():
        if isinstance(v, Namespace):
            cfg_dict[k] = to_dict(v)
        elif is_dataclass(v):
            cfg_dict[k] = to_dict(v)
        elif isinstance(v, Path):
            cfg_dict[k] = str(v)
        else:
            cfg_dict[k] = v

    return cfg_dict


def save_config(cfg, path):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)


def load_from_dict(cfg, dict_config):
    for k, v in dict_config.items():
        if is_dataclass(cfg.__dict__[k]):
            cfg.__dict__[k] = load_from_dict(cfg.__dict__[k], v)
        else:
            cfg.__dict__[k] = v

    return cfg


def load_config(path) -> Config:
    cfg = Config()
    with open(path, 'r') as f:
        dict_cfg = json.load(f)

    cfg = load_from_dict(cfg, dict_cfg)

    return cfg


CONFIG_PATH = 'config.json'


class UtilsTest(unittest.TestCase):

    def tearDown(self):
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)

        assert not os.path.exists(CONFIG_PATH)

    def test_load_config(self):
        cfg = Config()
        self.assertEqual(cfg.DATA.TEMPORAL_RANDOM_OFFSET, 0)
        self.assertEqual(cfg.DATA.TEMPORAL_RANDOM_JITTER, 0)
        self.assertEqual(cfg.TRAIN.RESULT_DIR, Path('results'))
        cfg.DATA.TEMPORAL_RANDOM_OFFSET = 24
        cfg.DATA.TEMPORAL_RANDOM_JITTER = 7
        cfg.TRAIN.RESULT_DIR = 'results/123'

        dict_cfg = to_dict(cfg)
        save_config(dict_cfg, CONFIG_PATH)
        self.assertTrue(os.path.exists(CONFIG_PATH))
        cfg = load_config(CONFIG_PATH)
        self.assertEqual(cfg.DATA.TEMPORAL_RANDOM_OFFSET, 24)
        self.assertEqual(cfg.DATA.TEMPORAL_RANDOM_JITTER, 7)
        self.assertEqual(cfg.TRAIN.RESULT_DIR, 'results/123')
