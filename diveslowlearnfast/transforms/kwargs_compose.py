from torchvision.transforms.v2 import Compose

class KwargsCompose(Compose):

    def __init__(self, transforms, **kwargs):
        super().__init__(transforms)
        self.accepts_kwargs = True


    def __call__(self, x, **kwargs):
        for t in self.transforms:
            if hasattr(t, 'accepts_kwargs') and t.accepts_kwargs:
                x = t(x, **kwargs)
            else:
                x = t(x)
        return x