import unittest
import torch

from diveslowlearnfast.models import SlowFast
from diveslowlearnfast.config.defaults import defaults as cfg

class SlowFastTest(unittest.TestCase):
    def test_should_forward(self):
        model = SlowFast(cfg)
        # B x T x C x W x H
        x = torch.randn(2, 4, 3, 224, 224)
        o = model(x)
        self.assertEqual(o.size, torch.Size([48]))

if __name__ == '__main__':

    unittest.main()
