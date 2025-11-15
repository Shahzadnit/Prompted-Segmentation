from typing import List

import torch
import torch.nn as nn
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import numpy as np


class CLIPSegFineTuner(nn.Module):
    """
    Wrapper around HuggingFace CLIPSeg for image+text segmentation.
    Ensures returned logits have shape [B,1,H,W].
    """

    def __init__(self, model_name: str = "CIDAS/clipseg-rd64-refined", image_size: int = 224):
        super().__init__()
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)

        # Fix processor image size to desired resolution
        self.processor.image_processor.size["height"] = image_size
        self.processor.image_processor.size["width"] = image_size

    def forward(self, pixel_values: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        pixel_values: [B,3,H,W] float in [0,1]
        prompts: list of B strings
        Returns:
          logits: [B,1,H,W]
        """
        device = next(self.parameters()).device

        images_pil = []
        for img in pixel_values:  # [3,H,W]
            img_np = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))  # H,W,C
            images_pil.append(Image.fromarray(img_np))

        inputs = self.processor(
            text=prompts,
            images=images_pil,
            padding="max_length",
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        logits = outputs.logits  # [B,H,W] for CLIPSeg
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)  # [B,1,H,W]

        return logits
