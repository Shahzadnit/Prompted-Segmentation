import torch
def compute_iou_dice_from_logits(logits: torch.Tensor, masks: torch.Tensor, thresh: float = 0.5):
    """
    logits: [B,1,H,W]
    masks:  [B,1,H,W] (0/1)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)

    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    dice = (2 * intersection + 1e-6) / (preds.sum(dim=1) + masks.sum(dim=1) + 1e-6)

    return iou.mean().item(), dice.mean().item()
