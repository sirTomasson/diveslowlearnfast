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

    def forward(self, inputs, y=None, **kwargs):
        if y is not None:
            y = y.to(self.device)

        localisation_maps, logits = self.gradcam(inputs, y)
        return localisation_maps, logits

class ExplainerStrategy:

    @staticmethod
    def get_explainer(model: nn.Module,
                      cfg: Config,
                      device: torch.device) -> nn.Module:

        assert cfg.EGL.METHOD in ['gradcam']
        if cfg.EGL.METHOD == 'gradcam':
            return GradCamExplainer(model, cfg=cfg, device=device)

        raise ValueError('Unsupported method')

