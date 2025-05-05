
import random

from torchvision.transforms.v2.functional import hflip
from torchvision.transforms.v2 import Transform

class DeterministicHorizontalFlip(Transform):

    def __init__(self, p=0.5):
        super(self.__class__, self).__init__()
        self.p = p
        self.accepts_kwargs = True


    def forward(self, inputs, flip=None, **kwargs):
        if flip is None:
            flip = random.random() > 0.5

        return hflip(inputs) if flip else inputs