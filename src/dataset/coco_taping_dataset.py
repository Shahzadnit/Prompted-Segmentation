import os
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as T


class TapingCocoDataset(Dataset):
    """
    Dataset 1: Drywall-Join-Detect (taping area) with only bounding boxes.
    We convert each bbox into a rectangular segmentation mask.

    Returns:
      image: [3,H,W] float tensor
      mask:  [1,H,W] float 0/1
      prompt: "segment taping area"
    """

    def __init__(
        self,
        images_dir: str,
        ann_path: str,
        prompt: str = "segment taping area",
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.images_dir = images_dir
        self.prompt = prompt
        self.image_size = image_size

        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.imgid_to_anns: Dict[int, List[dict]] = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.imgid_to_anns.setdefault(img_id, []).append(ann)

        self.id_to_img = {img["id"]: img for img in self.images}

        # Typically category_id=1 for taping area; adjust if needed
        self.taping_cat_ids = {1}

        self.img_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.BILINEAR),
            T.ToTensor(),
        ])

        self.mask_resize = T.Resize(self.image_size, interpolation=Image.NEAREST)
        self.image_ids = [img["id"] for img in self.images]

    def __len__(self):
        return len(self.image_ids)

    def _bboxes_to_mask(self, bboxes, height, width):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for (x, y, w, h) in bboxes:
            x0, y0 = x, y
            x1, y1 = x + w, y + h
            draw.rectangle([x0, y0, x1, y1], outline=1, fill=1)
        return mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.id_to_img[img_id]
        file_name = img_info["file_name"]
        H, W = img_info["height"], img_info["width"]

        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        anns = self.imgid_to_anns.get(img_id, [])
        bboxes = []
        for ann in anns:
            if ann["category_id"] not in self.taping_cat_ids:
                continue
            x, y, w, h = ann["bbox"]
            bboxes.append((x, y, w, h))

        mask = self._bboxes_to_mask(bboxes, H, W)
        image = self.img_transform(image)
        mask = self.mask_resize(mask)

        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask_tensor,
            "prompt": self.prompt,
            "file_name": file_name,
            "image_id": img_id,
            "orig_size": (H, W),
        }
