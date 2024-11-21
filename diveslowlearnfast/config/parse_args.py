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

def include_config_in_parser(cfg, parser, namespace=None):
    cfg_dict = cfg.__dict__
    for k, v in cfg_dict.items():
        if is_dataclass(v):
            include_config_in_parser(v, parser, namespace=k)
        else:
            arg = f'{namespace}.{k}' if namespace else k
            if type(v) is bool:
                parser.add_argument(f"--{arg}", default=v, action='store_true')
            else:
                parser.add_argument(f"--{arg}", type=type(v), default=v)


def parse_args() -> Config:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SlowFast network runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'DATA.ANNOTATIONS_PATH',
        help='Path to the annotations file'
    )

    parser.add_argument(
        'DATA.VOCAB_PATH',
        help='Path to the vocabulary file'
    )

    parser.add_argument(
        'DATA.VIDEOS_PATH',
        help='Path to the video files'
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
        args = parse_args('xyz', 'abc', 'foo',
                          '--DATA_LOADER.PIN_MEMORY',
                          '--DATA_LOADER.NUM_WORKERS', '2')
        self.assertEqual(args.DATA.ANNOTATIONS_PATH, 'xyz')
        self.assertEqual(args.DATA_LOADER.NUM_WORKERS, 2)
        self.assertTrue(args.DATA_LOADER.PIN_MEMORY)



