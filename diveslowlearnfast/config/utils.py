import json
from argparse import Namespace
from dataclasses import is_dataclass
from pathlib import Path

from diveslowlearnfast.config import Config

import copy

def to_dict(value):
    cfg_dict = value.__dict__
    for k, v in cfg_dict.items():
        if isinstance(v, Namespace):
            cfg_dict[k] = to_dict(v)
        elif isinstance(v, Path):
            cfg_dict[k] = str(v)
        else:
            cfg_dict[k] = v

    return cfg_dict

def save_config(cfg: Config, path):
    cfg = copy.deepcopy(cfg)
    cfg_dict = to_dict(cfg)
    print(cfg_dict)
    with open(path, 'w') as f:
        json.dump(cfg_dict, f, indent=2)