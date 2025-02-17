import torch

from torch import nn

from diveslowlearnfast.config import Config
from diveslowlearnfast.visualise.gradcam import GradCAM


class GradCamExplainer(nn.Module):

    def __init__(self, model, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.gradcam = GradCAM(model,
                               cfg.GRADCAM.TARGET_LAYERS,
                               cfg.DATA.MEAN,
                               cfg.DATA.STD,
                               cfg.GRADCAM.COLORMAP)

    def forward(self, x, y=None, **kwargs):
        xb_fast = x[:].to(self.device)
        # reduce the number of frames by the alpha ratio
        # B x C x T / alpha x H x W
        xb_slow = x[:, :, ::self.cfg.SLOWFAST.ALPHA].to(self.device)
        if y is not None:
            y = y.to(self.device)
        _, localisation_maps, _ = self.gradcam([xb_slow, xb_fast], y)
        return localisation_maps[1] # only return the xb_fast explanation.

class ExplainerStrategy:

    @staticmethod
    def get_explainer(model: nn.Module,
                      cfg: Config,
                      device: torch.device) -> nn.Module:

        assert cfg.EGL.METHOD in ['gradcam']
        if cfg.EGL.METHOD == 'gradcam':
            return GradCamExplainer(model, cfg=cfg, device=device)

        raise ValueError('Unsupported method')

