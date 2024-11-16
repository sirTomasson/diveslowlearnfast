from pytorchvideo.transforms.functional import uniform_crop
from torchvision.transforms.v2 import Transform


class CenterCropVideo(Transform):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, x):
        # 1 for center crop
        return uniform_crop(x, self.size, spatial_idx=1)