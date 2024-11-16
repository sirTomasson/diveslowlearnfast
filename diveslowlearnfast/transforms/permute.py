from torchvision.transforms.v2 import Transform


class Permute(Transform):

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def __call__(self, x):
        return x.permute(*self.shape)