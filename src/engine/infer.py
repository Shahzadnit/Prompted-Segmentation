import os
import re
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image


from src.utils.helpers import ensure_dir


@torch.no_grad()
def run_inference(
    model,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    prompt_str: str,
    return_masks: bool = False,
    collect_gt: bool = False,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Run inference for a single prompt_str.

    Saves individual PNG masks to `save_dir` and (optionally) returns:

      all_masks: {file_name: {prompt_str: mask_np}}
      gt_masks:  {file_name: gt_mask_np}   (only if collect_gt=True)

    mask_np and gt_mask_np are uint8 HxW arrays in {0,255}.
    """
    ensure_dir(save_dir)
    model.eval()

    all_masks: Dict[str, Dict[str, np.ndarray]] = {}
    gt_masks: Dict[str, np.ndarray] = {}

    for batch in tqdm(loader, desc=f"Infer ({prompt_str})", leave=False):
        images = batch["image"].to(device)
        file_names: List[str] = batch["file_name"]
        orig_sizes = batch["orig_size"]
        prompts = [prompt_str] * images.size(0)

        logits = model(images, prompts)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()  # [B,1,h,w]

        # Normalize orig_sizes into Hs and Ws
        if isinstance(orig_sizes, torch.Tensor):
            Hs = orig_sizes[:, 0]
            Ws = orig_sizes[:, 1]
        elif isinstance(orig_sizes, (list, tuple)):
            if len(orig_sizes) == 2 and isinstance(orig_sizes[0], torch.Tensor):
                Hs, Ws = orig_sizes
            else:
                Hs = [s[0] for s in orig_sizes]
                Ws = [s[1] for s in orig_sizes]
        else:
            s = torch.as_tensor(orig_sizes)
            Hs, Ws = s[:, 0], s[:, 1]

        # Optional GT collection
        if collect_gt:
            gt_batch = batch["mask"]  # [B,1,h,w] at training res

        for i in range(images.size(0)):
            pred = preds[i : i + 1]

            if isinstance(Hs, torch.Tensor):
                H_orig = int(Hs[i].item())
                W_orig = int(Ws[i].item())
            else:
                H_orig = int(Hs[i])
                W_orig = int(Ws[i])

            # Upsample prediction to original size
            pred_up = F.interpolate(pred, size=(H_orig, W_orig), mode="nearest")
            mask_np = pred_up.squeeze().cpu().numpy()  # [H,W] in {0,1}
            mask_np = (mask_np * 255).astype(np.uint8)

            base_name = os.path.splitext(file_names[i])[0]
            prompt_tag = re.sub(r"[^A-Za-z0-9]+", "_", prompt_str)  # filesystem-safe
            out_name = f"{base_name}__{prompt_tag}.png"
            out_path = os.path.join(save_dir, out_name)
            Image.fromarray(mask_np).save(out_path)


            if return_masks:
                all_masks.setdefault(file_names[i], {})[prompt_str] = mask_np

            if collect_gt:
                # Upsample GT mask to original size as well
                gt = gt_batch[i : i + 1]  # [1,1,h,w]
                gt_up = F.interpolate(gt.float(), size=(H_orig, W_orig), mode="nearest")
                gt_np = gt_up.squeeze().cpu().numpy()  # [H,W] in {0,1}
                gt_np = (gt_np * 255).astype(np.uint8)
                gt_masks[file_names[i]] = gt_np

    if return_masks and collect_gt:
        return all_masks, gt_masks
    if return_masks:
        return all_masks, {}
    return {}, {}
