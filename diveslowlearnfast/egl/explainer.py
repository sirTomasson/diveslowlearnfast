import torch

from torch import nn

from diveslowlearnfast.config import Config
from diveslowlearnfast.visualise.gradcam import GradCAM
from diveslowlearnfast.datasets import superimpose_confounder

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
                      device: torch.device):

        assert cfg.EGL.METHOD in ['gradcam', 'confounder']
        if cfg.EGL.METHOD == 'gradcam':
            return GradCamExplainer(model, cfg=cfg, device=device)

        elif cfg.EGL.METHOD == 'confounder':
            def _confounder_explainer(inputs, yb):
                result = []
                for inp in inputs:
                    sub_result = []
                    for x, y in zip(inp, yb):
                        x = torch.zeros_like(x)
                        x = superimpose_confounder(x, y, inplace=True)
                        x = torch.mean(x, dim=0, keepdim=True)
                        sub_result.append(x)

                    result.append(torch.stack(sub_result))

                return result, model(inputs)

            return _confounder_explainer

        raise ValueError('Unsupported method')

