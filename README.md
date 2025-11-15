# Prompted Segmentation for Drywall QA

This repository implements text-conditioned segmentation for:

- **Dataset 1 (Taping area)** – prompt: `"segment taping area"`
- **Dataset 2 (Cracks)** – prompt: `"segment crack"`

using a fine-tuned **CLIPSeg** model.

The code trains on 224×224 images/masks (CLIP standard input size) and upsamples
predicted masks back to **640×640** for submission, matching original image size.

---

## 1. Requirements

```bash
pip install -r requirements.txt
