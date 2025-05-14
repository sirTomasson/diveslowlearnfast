import torch

import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, ce_loss, smooth=1e-8, alpha=1.0, beta=1.0):
        super().__init__()
        self.smooth = smooth
        self.ce_loss = ce_loss
        self.alpha = alpha
        self.beta = beta

    def forward(self, targets, logits, gradcams, masks):
        batch_size = gradcams[0].size(0)
        dice_loss = 0.0
        for gradcam, mask in zip(gradcams, masks):
            gradcam_flat = gradcam.reshape(batch_size, -1)
            mask_flat = mask.reshape(batch_size, -1)

            intersection = torch.min(gradcam_flat, mask_flat).sum(dim=1)
            predictions_sum = gradcam_flat.sum(dim=1)
            targets_sum = mask_flat.sum(dim=1)

            dice = (2.0 * intersection + self.smooth) / (predictions_sum + targets_sum + self.smooth)
            dice_loss += ((1.0 - dice.mean()) / len(gradcams))

        ce_loss = self.alpha * self.ce_loss(logits, targets)
        dice_loss *= self.beta
        total_loss = ce_loss + dice_loss
        return total_loss, { 'ce_loss': ce_loss.item(), 'dice_loss': dice_loss.item(), 'total_loss': total_loss.item() }