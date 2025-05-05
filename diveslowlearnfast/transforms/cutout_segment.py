import os
import random
from diveslowlearnfast.datasets.utils import read_diver_segmentation_mask
from torchvision.transforms.v2 import Transform


class CutoutSegment(Transform):

    def __init__(self, dataset_path, p=0.5):
        super().__init__()
        self.dataset_path = dataset_path
        self.accepts_kwargs = True
        self.p = p

    def __call__(self, x, vidname=None, indices=None, **kwargs):
        assert vidname is not None
        path = os.path.join(self.dataset_path, vidname)
        masks = read_diver_segmentation_mask(path, indices=indices)
        if random.random() > self.p:
            return x

        return x * masks[:, :, :, None]