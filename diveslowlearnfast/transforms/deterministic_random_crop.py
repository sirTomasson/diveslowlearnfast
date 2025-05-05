
import torch

from torchvision.transforms.v2 import Transform, RandomCrop
from torchvision.transforms.v2.functional import crop

class DeterministicRandomCrop(Transform):

    def __init__(self, size):
        super(self.__class__, self).__init__()
        self.size = size
        self.accepts_kwargs = True

    def forward(self, inputs, crop_params=None, **kwargs):
        h, w = inputs.shape[:-2]
        if crop_params is None:
            crop_params = RandomCrop.get_params(torch.zeros((h, w)), output_size=self.size)

        return crop(inputs, *crop_params)