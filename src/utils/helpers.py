# import os
# import random
# import argparse

# import numpy as np
# import torch


# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Prompted Segmentation for Drywall QA")

#     parser.add_argument(
#         "--mode",
#         type=str,
#         required=True,
#         choices=["train", "eval", "infer", "full"],
#         help="train / eval / infer / full (full = train+eval+infer)",
#     )

#     parser.add_argument(
#         "--dataset",
#         type=str,
#         required=True,
#         choices=["cracks", "taping"],
#         help="which dataset to use",
#     )

#     parser.add_argument(
#         "--data_root",
#         type=str,
#         required=True,
#         help="path to /media/sdb_access/Assignment",
#     )

#     parser.add_argument(
#         "--output_root",
#         type=str,
#         default="./outputs",
#         help="root directory for checkpoints, logs, predictions",
#     )

#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help="batch size for training/eval/infer",
#     )

#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=10,
#         help="number of training epochs (used in train/full modes)",
#     )

#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=1e-5,
#         help="learning rate",
#     )

#     parser.add_argument(
#         "--num_workers",
#         type=int,
#         default=4,
#         help="DataLoader workers",
#     )

#     parser.add_argument(
#         "--checkpoint",
#         type=str,
#         default="",
#         help="path to checkpoint for eval/infer (if empty for eval, uses best)",
#     )

#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda",
#         help="cuda or cpu",
#     )

#     parser.add_argument(
#         "--image_size",
#         type=int,
#         default=224,
#         help="training image size (square), default 224",
#     )

#     return parser.parse_args()


# def get_device(device_str: str) -> torch.device:
#     if device_str == "cuda" and not torch.cuda.is_available():
#         print("CUDA not available, falling back to CPU.")
#         return torch.device("cpu")
#     return torch.device(device_str)


import os
import random
import argparse

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompted Segmentation for Drywall QA")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "infer", "full"],
        help="train / eval / infer / full (full = train+eval+infer)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cracks", "taping"],
        help="which dataset to use",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="path to /media/sdb_access/Assignment",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs",
        help="root directory for checkpoints, logs, predictions, reports",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size for training/eval/infer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of training epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="path to checkpoint for eval/infer (if empty for eval, uses best)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="training image size (square), default 224",
    )

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)
