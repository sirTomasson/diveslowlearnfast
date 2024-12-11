import argparse
import unittest
from dataclasses import is_dataclass

from diveslowlearnfast.config.defaults import ConfigData, Config


def dict_to_namespace(d) -> Config:
    """Recursively convert a dictionary to a namespace."""
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def int_list(comma_seperated):
    return [int(x) for x in comma_seperated.split(',')]

def as_list(comma_seperated):
    return [x for x in comma_seperated.split(',')]

def include_config_in_parser(cfg, parser, namespace=None):
    cfg_dict = cfg.__dict__
    for k, v in cfg_dict.items():
        if is_dataclass(v):
            include_config_in_parser(v, parser, namespace=k)
        else:
            arg = f'{namespace}.{k}' if namespace else k
            if type(v) is bool:
                parser.add_argument(f'--{arg}', default=v, action='store_true')
            elif type(v) is list:
                if len(v) > 0 and type(v[0]) is int:
                    _as_list = int_list
                else:
                    _as_list = as_list
                parser.add_argument(f'--{arg}', default=v, type=_as_list)
            else:
                parser.add_argument(f"--{arg}", type=type(v), default=v)


def parse_args() -> Config:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SlowFast network runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'DATA.DATASET_PATH',
        help='Path to the dataset root file. Should contain and rgb/ folder annotation files for the test and train set, and a vocabulary file.'
    )

    include_config_in_parser(Config(), parser)

    args = parser.parse_args()
    nested_dict = {}
    for arg, value in vars(args).items():
        parts = arg.split('.')
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return dict_to_namespace(nested_dict)


class ParseArgsTest(unittest.TestCase):

    def test_parse_args(self):
        args = parse_args('xyz',
                          '--DATA_LOADER.PIN_MEMORY',
                          '--DATA_LOADER.NUM_WORKERS', '2',
                          '--SOLVER.STEPS', '1,2,3,4')
        self.assertEqual(args.DATA_LOADER.NUM_WORKERS, 2)
        self.assertTrue(args.DATA_LOADER.PIN_MEMORY)
        self.assertEqual(args.SOLVER.STEPS, [1, 2, 3, 4])



