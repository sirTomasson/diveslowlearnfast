import math
import torch

from torchvision.transforms.v2 import Transform


def get_short_side_scale_params(h, w, min_size, max_size):
    size = torch.randint(min_size, max_size + 1, (1,)).item()
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return new_h, new_w


class DeterministicRandomShortSideScale(Transform):

    def __init__(self, min_size=256, max_size=320, interpolation='bilinear'):
        super(self.__class__, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation
        self.accepts_kwargs = True

    def forward(self, inputs, short_side_scale_params=None, **kwargs):
        if short_side_scale_params is None:
            h, w = inputs.size()[-2:]
            new_h, new_w = get_short_side_scale_params(
                h, w, self.min_size, self.max_size
            )
        else:
            new_h, new_w = short_side_scale_params

        return torch.nn.functional.interpolate(
            inputs, size=(new_h, new_w), mode=self.interpolation, align_corners=False
        )
