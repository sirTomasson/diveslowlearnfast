import json
import os

import numpy as np
import matplotlib.pyplot as plt

def main():
    dataset_path = '/home/s2871513/Datasets/Diving48/'
    train_labels = os.path.join(dataset_path, 'Diving48_V2_train.json')
    with open(train_labels, 'rb') as f:
        data = json.load(f)

    train_labels = os.path.join(dataset_path, 'Diving48_V2_test.json')
    with open(train_labels, 'rb') as f:
        test_data = json.load(f)

    plt.figure(figsize=(10, 6))

    data = list(map(lambda x: x['label'], data))
    data = np.array(data)
    test_data = list(map(lambda x: x['label'], test_data))
    data = np.array(data)

    unique, counts = np.unique(data, return_counts=True)
    _, test_counts = np.unique(test_data, return_counts=True)
    counts += test_counts

    mean = np.mean(counts)
    std = np.std(counts)
    plt.bar(unique, counts)
    plt.xticks(unique, unique, rotation=45, ha='right')
    plt.axhline(mean,  color='green', linestyle='--', label=f'$\mu$ = {mean:.2f}')
    plt.fill_between(unique,
                    mean - std,  # Lower bound
                    mean + std,  # Upper bound
                    color='red',
                    alpha=0.3,
                    label=f'±1 $\sigma$ = {std:.2f}')

    plt.margins(x=0)  # Remove horizontal padding
    plt.grid()
    plt.xlabel('class id')
    plt.ylabel('count')
    plt.legend()
    plt.savefig('../results/diving48_distribution.png')
    plt.show()


if __name__ == '__main__':
    main()