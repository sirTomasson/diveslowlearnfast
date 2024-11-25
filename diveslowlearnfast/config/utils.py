import json
from argparse import Namespace
from pathlib import Path

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

def save_config(cfg, path):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)