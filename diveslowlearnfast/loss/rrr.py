import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

logger = logging.getLogger(__name__)

class RRRLoss(nn.Module):
    def __init__(self, lambda1=1000.0):
        """
        Right for the Right Reasons loss function.

        Bibtex:
            @article{ross2017RRR,
              title={Right for the right reasons: Training differentiable models by constraining their explanations},
              author={Ross, Andrew Slavin and Hughes, Michael C and Doshi-Velez, Finale},
              journal={arXiv preprint arXiv:1703.03717},
              year={2017}
            }

        Args:
            lambda1: Weight for the explanation (input gradient) loss term
        """
        super().__init__()
        self.lambda1 = lambda1
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets, inputs, masks):
        """
        Compute the RRR loss.

        Args:
            logits: Model output logits (B x num_classes)
            targets: Ground truth labels (B)
            model: The model being trained
            inputs: Input data (B x ...)
            masks: Binary masks indicating where gradients should be small (B x ...)

        Returns:
            total_loss: Combined loss value
            losses: Dictionary containing individual loss components
        """
        ce_loss = self.cross_entropy(logits, targets)
        # boolean mask to select indices for which there is a mask available
        # this little optimisation ensures that we do not use autograd on inputs that will
        # be ignored anyway
        summed_log_probs = F.softmax(logits, dim=1).sum()
        gradients = torch.autograd.grad(summed_log_probs, inputs, create_graph=True, retain_graph=True)[0]
        gradient_loss = (masks * gradients).pow(2).mean()

        total_loss = ce_loss + self.lambda1 * gradient_loss

        losses = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'gradient_loss': gradient_loss.item()
        }

        return total_loss, losses


def _normalise_gradients(gradients):
    return gradients / (gradients.abs().mean() + 1e-7)


class DualPathRRRLoss(nn.Module):


    def __init__(self,
                 lambdas=None,
                 normalise_gradients=False,
                 force_new_gradients=False,
                 skip_zero_masks=False):
        super().__init__()
        if lambdas is None:
            lambdas = [0.5e12, 0.5e12]

        self.lambdas = lambdas
        self.normalise_gradients = normalise_gradients
        self.force_new_gradients = force_new_gradients
        self.skip_zero_masks = skip_zero_masks
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets, inputs, masks):
        ce_loss = self.cross_entropy(logits, targets)
        total_loss = ce_loss
        losses = {
            'ce_loss': ce_loss.item(),
        }

        relevant_indices = _get_binary_index_mask(masks)
        if self.skip_zero_masks and torch.sum(relevant_indices) == 0:
            losses['total_loss'] = total_loss.item()
            for idx in range(len(inputs)):
                losses[f'gradient_loss_path_{idx}'] = 0
            return total_loss, losses

        relevant_logits = logits[relevant_indices]
        summed_log_probs = F.log_softmax(relevant_logits, dim=1).sum()

        for idx, (inp, mask) in enumerate(zip(inputs, masks)):
            gradients = self._calculate_gradient(inp, summed_log_probs)

            if self.normalise_gradients:
                gradients = _normalise_gradients(gradients)
            gradient_loss = self.lambdas[idx] * (mask * gradients).pow(2).mean()  # Added lambda1 scaling
            total_loss += gradient_loss
            losses[f'gradient_loss_path_{idx}'] = gradient_loss.item()

        losses['total_loss'] = total_loss.item()
        return total_loss, losses


    def _calculate_gradient(self, inp, summed_log_probs):
        if inp.grad is None or self.force_new_gradients:
            logger.debug('No gradients on "inp", recalculating gradients using summed_log_probs')
            return torch.autograd.grad(
                summed_log_probs,
                inp,
                create_graph=True,
                retain_graph=True
            )[0]

        logger.debug('Gradients exist on "inp", reusing existing gradients')
        return inp.grad


def _get_binary_index_mask(masks):
    masks = masks[0] # the slow mask
    return torch.sum(masks, dim=(1, 2, 3, 4)) > 0