import time
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mask = (masks.sum(dim=(1, 2, 3, 4)) > 0)
        masked_elements = inputs[mask]

        if masked_elements.shape[0] > 0:
            log_probs = F.softmax(masked_elements, dim=1)
            summed_log_probs = log_probs.sum()
            gradients = torch.autograd.grad(summed_log_probs, inputs, create_graph=True, retain_graph=True)[0]
            gradient_loss = (masks * gradients).pow(2).mean()
            gradient_loss_item = gradient_loss.item()
        else:
            gradient_loss = 0
            gradient_loss_item = gradient_loss

        total_loss = ce_loss + self.lambda1 * gradient_loss

        losses = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'gradient_loss': gradient_loss_item
        }

        return total_loss, losses
