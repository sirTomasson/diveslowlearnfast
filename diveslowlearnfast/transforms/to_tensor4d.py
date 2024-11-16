import torch
from torchvision.transforms.v2 import Transform


class ToTensor4D(Transform):

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return torch.tensor(x)
