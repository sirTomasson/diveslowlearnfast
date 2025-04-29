import torch
from torchvision.transforms.v2 import Transform


class ToTensor4D(Transform):

    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.float32

    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype).permute(3, 0, 1, 2)
