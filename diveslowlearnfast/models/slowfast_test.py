import unittest
import torch

from diveslowlearnfast.config.defaults import Config
from diveslowlearnfast.models import SlowFast
from diveslowlearnfast.models.slowfast import VideoModelStem
import diveslowlearnfast.models.slowfast as sf


class SlowFastTest(unittest.TestCase):
    def test_should_forward_slowfast(self):
        cfg = Config()
        model = SlowFast(cfg)

        batch_size = 8
        slow_frames = 4
        fast_frames = 32
        height = 224
        width = 224

        x_slow = torch.randn(batch_size, 3, slow_frames, height, width)
        x_fast = torch.randn(batch_size, 3, fast_frames, height, width)
        x = [x_slow, x_fast]

        o = model(x)
        self.assertEqual(o.size(), torch.Size([8, 48]))

    def test_should_forward_stem(self):
        cfg = Config()

        temp_kernel = [
            [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
        ]

        model = VideoModelStem(
            dim_in=[3, 3],
            dim_out=[64, 64 // 8],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=sf.get_norm(cfg),
        )
        batch_size = 8
        slow_frames = 4
        fast_frames = 32
        height = 224
        width = 224

        x_slow = torch.randn(batch_size, 3, slow_frames, height, width)
        x_fast = torch.randn(batch_size, 3, fast_frames, height, width)
        x = [x_slow, x_fast]
        o = model(x)
        self.assertEqual(len(o), 2)
        self.assertEqual(o[0].shape, (batch_size, 64, 4, height//4, width//4))
        self.assertEqual(o[1].shape, (batch_size, 8, 32, height//4, width//4))


if __name__ == '__main__':
    unittest.main()
