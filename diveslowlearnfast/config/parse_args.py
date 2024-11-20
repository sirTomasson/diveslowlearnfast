import argparse

def dict_to_namespace(d):
    """Recursively convert a dictionary to a namespace."""
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SlowFast network runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'data.annotations_path',
        help='Path to the annotations file'
    )

    parser.add_argument(
        'data.vocab_path',
        help='Path to the vocabulary file'
    )

    parser.add_argument(
        'data.videos_path',
        help='Path to the video files'
    )

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