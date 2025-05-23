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
        self.model = GradCAM(model,
                             cfg.GRADCAM.TARGET_LAYERS,
                             cfg.DATA.MEAN,
                             cfg.DATA.STD,
                             cfg.GRADCAM.COLORMAP)

    def forward(self, inputs, y=None, **kwargs):
        if y is not None:
            y = y.to(self.device)

        return self.model(inputs, y)


class ConfounderExplainer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, inputs, y=None, **kwargs):
        preds = self.model(inputs)
        if not self.model.training:
            return preds

        result = []
        for inp in inputs:
            sub_result = []
            for x, y in zip(inp, y):
                x = torch.zeros_like(x)
                x = superimpose_confounder(x, y, inplace=True)
                x = torch.mean(x, dim=0, keepdim=True)
                sub_result.append(x)

            result.append(torch.stack(sub_result))

        return result, preds


class NoopExplainer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, **kwargs):
        return None, self.model(inputs)

class ExplainerStrategy:

    @staticmethod
    def get_explainer(model: nn.Module,
                      cfg: Config,
                      device: torch.device):

        assert cfg.EGL.METHOD in ['gradcam', 'confounder', 'ogl', 'cache']
        if cfg.EGL.METHOD in ['gradcam', 'ogl']:
            return GradCamExplainer(model, cfg=cfg, device=device)

        elif cfg.EGL.METHOD == 'confounder':
            return ConfounderExplainer(model)

        return NoopExplainer(model)

