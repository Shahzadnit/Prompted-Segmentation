import os
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageChops
import torchvision.transforms as T


class CrackCocoSegDataset(Dataset):
    """
    Dataset 2: Cracks, with polygon segmentations (COCO format).
    Returns:
      image: [3,H,W] float tensor
      mask:  [1,H,W] float 0/1
      prompt: "segment crack"
    """

    def __init__(
        self,
        images_dir: str,
        ann_path: str,
        prompt: str = "segment crack",
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.images_dir = images_dir
        self.prompt = prompt
        self.image_size = image_size

        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # Map image_id -> annotations
        self.imgid_to_anns: Dict[int, List[dict]] = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.imgid_to_anns.setdefault(img_id, []).append(ann)

        # Map image_id -> image info
        self.id_to_img = {img["id"]: img for img in self.images}

        # Category ids: treat all categories except "object" (id=2) as crack
        # Adjust if needed according to your JSON
        self.non_crack_cat_ids_to_ignore = {2}

        self.img_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.BILINEAR),
            T.ToTensor(),  # [0,1]
        ])

        self.mask_resize = T.Resize(self.image_size, interpolation=Image.NEAREST)
        self.image_ids = [img["id"] for img in self.images]

    def __len__(self):
        return len(self.image_ids)

    def _polygons_to_mask(self, polygons, height, width):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for poly in polygons:
            if len(poly) < 6:
                continue
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(xy, outline=1, fill=1)
        return mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.id_to_img[img_id]
        file_name = img_info["file_name"]
        H, W = img_info["height"], img_info["width"]

        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        anns = self.imgid_to_anns.get(img_id, [])
        mask = Image.new("L", (W, H), 0)

        for ann in anns:
            if ann["category_id"] in self.non_crack_cat_ids_to_ignore:
                continue
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            ann_mask = self._polygons_to_mask(seg, H, W)
            mask = ImageChops.lighter(mask, ann_mask)  # union

        image = self.img_transform(image)
        mask = self.mask_resize(mask)

        mask_np = np.array(mask, dtype=np.uint8)  # 0/1
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask_tensor,
            "prompt": self.prompt,
            "file_name": file_name,
            "image_id": img_id,
            "orig_size": (H, W),
        }
