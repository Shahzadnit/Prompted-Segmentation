from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_iou_dice_from_logits


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    prompt_str: str,
) -> float:
    """
    Train for one epoch using a single canonical prompt string.
    """
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc=f"Train ({prompt_str})", leave=False):
        images = batch["image"].to(device)   # [B,3,H,W]
        masks = batch["mask"].to(device)     # [B,1,H,W]
        prompts = [prompt_str] * images.size(0)

        logits = model(images, prompts)      # [B,1,h,w]

        if logits.shape[-2:] != masks.shape[-2:]:
            masks_resized = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")
        else:
            masks_resized = masks

        loss = F.binary_cross_entropy_with_logits(logits, masks_resized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b = images.size(0)
        total_loss += loss.item() * b
        n += b

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_one_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    prompt_str: str,
) -> Tuple[float, float, float]:
    """
    Evaluate model on one prompt string (mIoU & Dice).
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n = 0

    for batch in tqdm(loader, desc=f"Val ({prompt_str})", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        prompts = [prompt_str] * images.size(0)

        logits = model(images, prompts)

        if logits.shape[-2:] != masks.shape[-2:]:
            masks_resized = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")
        else:
            masks_resized = masks

        loss = F.binary_cross_entropy_with_logits(logits, masks_resized)
        iou, dice = compute_iou_dice_from_logits(logits, masks_resized)

        b = images.size(0)
        total_loss += loss.item() * b
        total_iou += iou * b
        total_dice += dice * b
        n += b

    return (
        total_loss / max(n, 1),
        total_iou / max(n, 1),
        total_dice / max(n, 1),
    )
