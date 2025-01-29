import torch
import torch.nn as nn
import torch.nn.functional as F


class RRRLoss(nn.Module):
    def __init__(self, lambda1=1000.0, lambda2=0.0001):
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
            lambda2: Weight for the L2 regularization term
        """
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets, model, inputs, masks):
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
        batch_size = logits.size(0)

        # Right answers loss (cross entropy)
        ce_loss = self.cross_entropy(logits, targets)

        # Right reasons loss (input gradient penalty)
        log_probs = F.log_softmax(logits, dim=1)

        # Sum across classes for each sample
        summed_log_probs = log_probs.sum(dim=1)

        # Compute input gradients
        gradients = torch.autograd.grad(
            summed_log_probs.sum(),
            inputs,
            create_graph=True,
            retain_graph=True
        )[0]

        # Apply mask and compute mean squared gradients
        masked_gradients = gradients * masks
        gradient_loss = (masked_gradients ** 2).mean()

        # L2 regularization on model parameters
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)

        # Combine losses
        total_loss = ce_loss + self.lambda1 * gradient_loss + self.lambda2 * l2_loss

        # Store individual loss components for logging
        losses = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'gradient_loss': gradient_loss.item(),
            'l2_loss': l2_loss.item()
        }

        return total_loss, losses