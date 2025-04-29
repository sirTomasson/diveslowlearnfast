import random

from torchvision.transforms.v2 import Transform

class RandomApply(Transform):

    def __init__(self, transforms, p=0.5, **kwargs):
        super().__init__()
        self.accepts_kwargs = True
        self.transforms = transforms
        self.p = p


    def __call__(self, x, **kwargs):
        idx = int(random.random() // self.p)
        t = self.transforms[idx]
        if hasattr(t, 'accepts_kwargs') and t.accepts_kwargs:
            x = t(x, **kwargs)
        else:
            x = t(x)

        return x