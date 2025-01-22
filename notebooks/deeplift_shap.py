import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple

import logging

logger = logging.getLogger(__name__)

class DeepLiftShap:
    def __init__(self, model: nn.Module):
        """
        Initialize DeepLiftShap algorithm

        Args:
            model: PyTorch neural network model
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode

    def attribute(self,
                  input_tensor: Tuple[torch.Tensor, torch.Tensor],
                  baselines: Tuple[torch.Tensor, torch.Tensor],
                  mode: str = 'pixel',
                  target = None,
                  n_samples: int = 100):
        """
        Compute attributions using DeepLiftShap

        Args:
            input_tensor: Input data point to explain
            target: Target class (for classification tasks)
            n_samples: Number of background samples to use

        Returns:
            Attribution scores for each input feature
        """
        # Randomly sample from background data
        logger.debug(f"Sampling {n_samples} background samples")

        # Compute scaled inputs
        logger.debug(f"Scaling input tensors and background samples using DeepLIFT rescale rule")
        baselines = self._sample_background(baselines, n_samples)
        scaled_inputs = self._get_scaled_inputs(input_tensor, baselines)

        # Compute attributions using DeepLIFT
        logger.debug(f"Computing DeepLIFT attributions")
        attributions = self._compute_deeplift_attributions(scaled_inputs, target)

        # Average attributions across samples, backgrounds, and channels
        return torch.mean(attributions, dim=[0, 1, 2])

    def _sample_background(self, background_data, n_samples: int) -> torch.Tensor:
        """Sample random background data points"""
        indices = np.random.choice(
            len(background_data),
            size=n_samples,
            replace=True
        )
        return background_data[indices]

    def _get_scaled_inputs(self,
                           input_tensor: torch.Tensor | List[torch.Tensor],
                           background_samples: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        """
        Scale inputs between background and input data point
        Uses the rescale rule from DeepLIFT
        """
        alphas = torch.linspace(0, 1, steps=5)
        scaled = torch.zeros(
            (len(alphas), len(background_samples)) + input_tensor.shape[1:]
        )

        for i, alpha in enumerate(alphas):
            scaled[i] = background_samples + alpha * (input_tensor - background_samples)

        return scaled

    def _compute_deeplift_attributions(self,
                                       scaled_input: torch.Tensor,
                                       target: Optional[int]) -> torch.Tensor:
        """
        Compute DeepLIFT attributions for scaled inputs
        """
        scaled_input.requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        scaled_inputs_reshaped = scaled_input.reshape(-1, *scaled_input.shape[2:])
        logger.debug(f'Computing model outputs, input type = {type(scaled_input)}')
        outputs = self.model(scaled_inputs_reshaped)

        if target is not None:
            outputs = outputs[:, target]
        else:
            outputs = torch.sum(outputs, dim=1)

        # Compute gradients
        logger.debug(f'Computing gradients for slow and fast input components')
        gradients = torch.autograd.grad(
            outputs.sum(),
            scaled_input,
            create_graph=True
        )[0]

        # Compute DeepLIFT attributions using the chain rule
        attributions = gradients * (scaled_input[-1] - scaled_input[0])

        return attributions

    def _compute_multipliers(self,
                             delta_out: torch.Tensor,
                             delta_in: torch.Tensor) -> torch.Tensor:
        """
        Compute multipliers using the RevealCancel rule from DeepLIFT
        """
        eps = 1e-10

        # Avoid division by zero
        delta_in_masked = torch.where(
            abs(delta_in) > eps,
            delta_in,
            torch.ones_like(delta_in) * eps
        )

        return delta_out / delta_in_masked

def main():
    from diveslowlearnfast.train import helper as train_helper
    from diveslowlearnfast.models import SlowFast, load_checkpoint
    from diveslowlearnfast.config import Config

    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)

    cfg = Config()
    cfg.DATA.DATASET_PATH = '/Users/youritomassen/Projects/xai/data/Diving48/'
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.DATA.NUM_FRAMES = 16
    device = torch.device('cpu')
    model = SlowFast(cfg)
    _, optimiser, *_ = train_helper.get_train_objects(cfg, model)
    model, *_ = load_checkpoint(model, optimiser, '../misc/checkpoint.pth', device)
    dl = DeepLiftShap(model)

    test_loader = train_helper.get_test_objects(cfg)
    x, y, *_ = next(iter(test_loader))
    x_fast = x[:]
    # reduce the number of frames by the alpha ratio
    # B x C x T / alpha x H x W
    B, C, T, H, W = x.shape
    x_slow = x[:, :, ::cfg.SLOWFAST.ALPHA]
    input = [x_slow, x_fast]

    # generate 100 baseline samples with black pixels
    baseline_values = torch.zeros((100, *x.shape[1:]))
    baseline_dist_fast = baseline_values[:]
    baseline_dist_slow = baseline_values[:, :, ::cfg.SLOWFAST.ALPHA]
    baselines = [baseline_dist_slow, baseline_dist_fast]
    attributions_slow, attributions_fast = dl.attribute(input, baselines, y, n_samples=5)

    print('attribution completed printing shapes for slow and fast input components:')
    print(attributions_slow.shape, attributions_fast.shape)
    alpha = cfg.SLOWFAST.ALPHA
    t, h, w = attributions_slow.shape
    slow_maps_upsampled = torch.zeros((t * alpha, H, W))
    for i in range(t):
        slow_maps_upsampled[i*alpha:(i+1)*alpha] = attributions_slow[i]

    # Combine pathways
    combined_maps = (slow_maps_upsampled + attributions_fast) / 2

    # Normalize
    max_val = torch.max(torch.abs(combined_maps))
    normalized_maps = combined_maps / (max_val + 1e-10)
    print(normalized_maps.shape)
    plt.matshow(normalized_maps.detach().cpu().numpy()[0])
    plt.show()


if __name__ == '__main__':
    main()