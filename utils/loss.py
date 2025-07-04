import torch
import torch.nn.functional as F

def batch_mask_loss(preds, targets, ce_loss_weight, dice_loss_weight):
    """
    CE loss + dice loss

    Args:
        preds: B,* logits
        targets: B,* binary
    Returns:
        loss: B
    """
    preds = preds.flatten(1)  ## B,-1
    targets = targets.flatten(1).to(preds)  ## B, -1
    sig_preds = torch.sigmoid(preds)  ## B, -1

    ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none").mean(dim=-1)  ## B
    dice_loss = 1.0 - ((2. * sig_preds * targets).sum(dim=-1)+1.0) / ((sig_preds+targets).sum(dim=-1) + 1.0)  ## B
    return ce_loss * ce_loss_weight + dice_loss * dice_loss_weight

def batch_mask_loss_in_points(preds, targets, ce_loss_weight, dice_loss_weight, K):
    """
    preds: *, H, W
    targets: *, H, W
    """
    H, W = preds.shape[-2::]
    
    if H*W <= K:
        return batch_mask_loss(preds, targets, ce_loss_weight, dice_loss_weight)
    
    assert targets.shape[-2::]==preds.shape[-2::]
    khi = torch.randint(low=0, high=H, size=(K,)).to(preds.device).long().reshape(-1)
    kwi = torch.randint(low=0, high=W, size=(K,)).to(preds.device).long().reshape(-1)
    return batch_mask_loss(
        preds=preds.reshape(-1, H, W)[:, khi, kwi],
        targets=targets.reshape(-1, H, W)[:, khi, kwi],
        ce_loss_weight=ce_loss_weight,
        dice_loss_weight=dice_loss_weight
    )
