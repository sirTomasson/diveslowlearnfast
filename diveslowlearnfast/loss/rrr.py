import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

logger = logging.getLogger(__name__)

class RRRLoss(nn.Module):
    def __init__(self,
                 lambdas=None,
                 normalise_gradients=False,
                 skip_zero_masks=False):
        super().__init__()

        if lambdas is None:
            lambdas = [0.01, 0.01]

        self.lambdas = lambdas
        self.normalise_gradients = normalise_gradients
        self.skip_zero_masks = skip_zero_masks

    def forward(self, logits, targets, inputs, masks, warmup=False):
        batch_size = logits.size(0)

        target_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        log_probs = F.log_softmax(logits, dim=1)

        right_answer_loss = -torch.sum(target_one_hot * log_probs) / batch_size

        total_loss = right_answer_loss
        losses = {'ce_loss': right_answer_loss.item()}

        if warmup or (self.skip_zero_masks and all(torch.sum(mask) == 0 for mask in masks)):
            losses['total_loss'] = total_loss.item()
            for idx in range(len(inputs)):
                losses[f'gradient_loss_path_{idx}'] = 0
            return total_loss, losses

        for idx, (inp, mask) in enumerate(zip(inputs, masks)):
            if torch.sum(mask) == 0 and self.skip_zero_masks:
                losses[f'gradient_loss_path_{idx}'] = 0
                continue

            gradients = torch.autograd.grad(
                log_probs,
                inp,
                grad_outputs=torch.ones_like(log_probs),
                create_graph=True,
                retain_graph=True
            )[0]

            if self.normalise_gradients:
                gradients = gradients / (torch.norm(gradients, dim=1, keepdim=True) + 1e-10)

            masked_gradients = mask * gradients

            n_frames = inp.size(2)
            l2_grad_loss = self.lambdas[idx] * torch.sum(masked_gradients**2) / (batch_size * n_frames)

            gradient_loss = l2_grad_loss
            total_loss += gradient_loss

            losses[f'gradient_loss_path_{idx}'] = gradient_loss.item()

        losses['total_loss'] = total_loss.item()
        return total_loss, losses