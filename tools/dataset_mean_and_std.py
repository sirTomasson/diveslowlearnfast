import argparse

import numpy as np
from tqdm import tqdm

from diveslowlearnfast.datasets import Diving48Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='path to dataset')
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = Diving48Dataset(args.dataset_path, -1, dataset_type='train')
    n = 0
    mean = np.zeros(3)
    M2 = np.zeros(3)

    dataset_iter = iter(dataset)
    progress_bar = tqdm(range(len(dataset)))
    for i in progress_bar:
        video, *_ = next(dataset_iter)
        video = video / 255
        video = video.reshape(-1, 3)
        n += len(video)
        delta = video - mean
        mean += np.mean(delta, axis=0)
        delta2 = video - mean
        M2 += np.sum(delta * delta2, axis=0)

        if (i+1) % 10 == 0:
            std = np.sqrt(M2 / (n - 1))
            progress_bar.set_postfix({
                'mu' : f'{mean}',
                'std': f'{std}'
            })

    std = np.sqrt(M2 / (n - 1))

    return mean, std


if __name__ == '__main__':
    mean, std = main()
    print(f'mean: {mean}, std: {std}')