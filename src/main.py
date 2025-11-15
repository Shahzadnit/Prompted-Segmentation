import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import re

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from src.utils.helpers import (
    parse_args,
    set_seed,
    get_device,
    ensure_dir,
)
from src.utils.report_utils import (
    write_train_report,
    write_prompt_metrics_report,
    write_inference_report,
)
from src.dataset.coco_crack_dataset import CrackCocoSegDataset
from src.dataset.coco_taping_dataset import TapingCocoDataset
from src.models.clipseg_finetune import CLIPSegFineTuner
from src.engine.train_eval import train_one_epoch, eval_one_epoch
from src.engine.infer import run_inference


# -------------------------------------------------------------------
# Prompt sets for each dataset
# -------------------------------------------------------------------

CRACK_PROMPTS = [
    "segment crack",
    "segment wall crack",
]

TAPING_PROMPTS = [
    "segment taping area",
    "segment joint/tape",
    "segment drywall seam",
]


def get_prompt_list(dataset_name: str):
    if dataset_name == "cracks":
        return CRACK_PROMPTS
    elif dataset_name == "taping":
        return TAPING_PROMPTS
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def prompt_to_tag(p: str) -> str:
    # Replace spaces, slashes, punctuation with underscores
    return re.sub(r"[^A-Za-z0-9]+", "_", p)



# -------------------------------------------------------------------
# Dataset builders
# -------------------------------------------------------------------

def build_datasets(args):
    image_size = (args.image_size, args.image_size)

    if args.dataset == "cracks":
        root = os.path.join(args.data_root, "Dataset_2")

        train_ds = CrackCocoSegDataset(
            images_dir=os.path.join(root, "train"),
            ann_path=os.path.join(root, "train_annotations.coco.json"),
            prompt="segment crack",
            image_size=image_size,
        )
        val_ds = CrackCocoSegDataset(
            images_dir=os.path.join(root, "valid"),
            ann_path=os.path.join(root, "valid_annotations.coco.json"),
            prompt="segment crack",
            image_size=image_size,
        )
        return train_ds, val_ds

    elif args.dataset == "taping":
        root = os.path.join(args.data_root, "Dataset_1")

        train_ds = TapingCocoDataset(
            images_dir=os.path.join(root, "train"),
            ann_path=os.path.join(root, "train_annotations.coco.json"),
            prompt="segment taping area",
            image_size=image_size,
        )
        val_ds = TapingCocoDataset(
            images_dir=os.path.join(root, "valid"),
            ann_path=os.path.join(root, "valid_annotations.coco.json"),
            prompt="segment taping area",
            image_size=image_size,
        )
        return train_ds, val_ds

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def build_test_dataset(args):
    """
    Only Dataset_2 (cracks) has a dedicated test split in your layout.
    For taping, we use valid split for inference.
    """
    if args.dataset != "cracks":
        return None

    root = os.path.join(args.data_root, "Dataset_2")
    test_ds = CrackCocoSegDataset(
        images_dir=os.path.join(root, "test"),
        ann_path=os.path.join(root, "test_annotations.coco.json"),
        prompt="segment crack",
        image_size=(args.image_size, args.image_size),
    )
    return test_ds


# -------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------

def annotate_tile(img: Image.Image, label: str) -> Image.Image:
    """
    Draw label text along the top border of the tile.
    """
    draw = ImageDraw.Draw(img)
    margin = 5
    text = label
    draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0))
    draw.text((margin, 2), text, fill=(255, 255, 255))
    return img


def build_stacked_visuals(
    images_dir: str,
    stacked_dir: str,
    prompt_list,
    all_masks_per_file,
    gt_masks,
):
    ensure_dir(stacked_dir)
    for fname, mask_dict in all_masks_per_file.items():
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            # skip if image missing
            continue
        orig_img = Image.open(img_path).convert("RGB")
        w, h = orig_img.size

        tiles = []
        labels = []

        # Input
        tiles.append(orig_img)
        labels.append("INPUT")

        # GT
        if fname in gt_masks:
            gt_np = gt_masks[fname]
            gt_img = Image.fromarray(gt_np, mode="L")
            gt_img = gt_img.resize((w, h))
            gt_rgb = Image.merge("RGB", (gt_img, gt_img, gt_img))
            tiles.append(gt_rgb)
            labels.append("GT")

        # Predictions per prompt
        for p in prompt_list:
            if p not in mask_dict:
                continue
            m = mask_dict[p]  # HxW uint8
            mask_img = Image.fromarray(m, mode="L")
            mask_img = mask_img.resize((w, h))
            mask_rgb = Image.merge("RGB", (mask_img, mask_img, mask_img))
            tiles.append(mask_rgb)
            labels.append(p)

        total_w = w * len(tiles)
        stacked = Image.new("RGB", (total_w, h))
        x = 0
        for tile, label in zip(tiles, labels):
            tile_annot = annotate_tile(tile.copy(), label)
            stacked.paste(tile_annot, (x, 0))
            x += w

        base_name = os.path.splitext(fname)[0]
        out_name = f"{base_name}__stacked.png"
        out_path = os.path.join(stacked_dir, out_name)
        stacked.save(out_path)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(42)
    device = get_device(args.device)

    ensure_dir(args.output_root)
    ckpt_dir = os.path.join(args.output_root, "checkpoints")
    ensure_dir(ckpt_dir)
    preds_root = os.path.join(args.output_root, "predictions")
    ensure_dir(preds_root)

    # Dataset-specific report directory, e.g. outputs/cracks or outputs/taping
    dataset_report_dir = os.path.join(args.output_root, args.dataset)
    ensure_dir(dataset_report_dir)

    # Build model
    model = CLIPSegFineTuner(image_size=args.image_size).to(device)

    # Helper: compute model size MB
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = num_params * 4 / (1024 ** 2)

    # ------------------------------------------------------------
    # TRAIN MODE
    # ------------------------------------------------------------
    if args.mode == "train":
        train_ds, val_ds = build_datasets(args)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_train = len(train_ds)
        num_val = len(val_ds)
        num_test = len(build_test_dataset(args)) if args.dataset == "cracks" else 0

        prompt_list = get_prompt_list(args.dataset)
        train_prompt = prompt_list[0]  # canonical training prompt
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        epoch_times = []
        best_dice = 0.0

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            t0 = time.perf_counter()
            train_loss = train_one_epoch(
                model, train_loader, optimizer=optimizer, device=device, prompt_str=train_prompt
            )
            t1 = time.perf_counter()
            epoch_time = t1 - t0
            epoch_times.append(epoch_time)
            print(f"Train loss ({train_prompt}): {train_loss:.4f} | epoch time: {epoch_time:.3f} s")

            # Evaluate on all prompts for this epoch
            prompt_metrics_last_epoch = {}
            for p in prompt_list:
                val_loss, val_iou, val_dice = eval_one_epoch(
                    model, val_loader, device, prompt_str=p
                )
                print(
                    f"[Prompt: '{p}'] "
                    f"Val loss: {val_loss:.4f} | "
                    f"mIoU: {val_iou:.4f} | Dice: {val_dice:.4f}"
                )
                prompt_metrics_last_epoch[p] = {
                    "loss": val_loss,
                    "miou": val_iou,
                    "dice": val_dice,
                }
                if val_dice > best_dice:
                    best_dice = val_dice
                    ckpt_path = os.path.join(ckpt_dir, f"{args.dataset}_best.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved best checkpoint to: {ckpt_path}")

        total_train_time = sum(epoch_times)

        # Write reports
        write_train_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            epoch_times=epoch_times,
            total_train_time=total_train_time,
            model_size_mb=model_size_mb,
        )
        write_prompt_metrics_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            prompt_metrics=prompt_metrics_last_epoch,
            split_name="val_last_epoch",
        )

    # ------------------------------------------------------------
    # EVAL MODE
    # ------------------------------------------------------------
    elif args.mode == "eval":
        if not args.checkpoint:
            args.checkpoint = os.path.join(ckpt_dir, f"{args.dataset}_best.pth")

        print(f"Loading checkpoint from: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

        _, val_ds = build_datasets(args)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        prompt_list = get_prompt_list(args.dataset)
        prompt_metrics = {}

        for p in prompt_list:
            val_loss, val_iou, val_dice = eval_one_epoch(
                model, val_loader, device, prompt_str=p
            )
            print(
                f"[Prompt: '{p}'] "
                f"Val loss: {val_loss:.4f} | "
                f"mIoU: {val_iou:.4f} | Dice: {val_dice:.4f}"
            )
            prompt_metrics[p] = {"loss": val_loss, "miou": val_iou, "dice": val_dice}

        write_prompt_metrics_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            prompt_metrics=prompt_metrics,
            split_name="val_eval",
        )

    # ------------------------------------------------------------
    # INFER MODE (multi-prompt + stacked)
    # ------------------------------------------------------------
    elif args.mode == "infer":
        if not args.checkpoint:
            raise ValueError("You must provide --checkpoint for infer mode.")

        print(f"Loading checkpoint from: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

        prompt_list = get_prompt_list(args.dataset)

        # Dataset for inference
        if args.dataset == "cracks":
            ds = build_test_dataset(args)
            if ds is None:
                raise ValueError("No test dataset defined for cracks.")
            base_pred_dir = os.path.join(preds_root, "cracks")
        else:  # taping
            _, val_ds = build_datasets(args)
            ds = val_ds
            base_pred_dir = os.path.join(preds_root, "taping")

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_images = len(ds)
        all_masks_per_file = {}
        gt_masks = {}
        t0 = time.perf_counter()

        for idx, p in enumerate(prompt_list):
            save_dir = os.path.join(base_pred_dir, prompt_to_tag(p))
            # Collect GT only once (idx == 0)
            collect_gt = idx == 0
            masks_for_p, gt_dict = run_inference(
                model,
                loader,
                device,
                save_dir,
                prompt_str=p,
                return_masks=True,
                collect_gt=collect_gt,
            )
            for fname, d in masks_for_p.items():
                all_masks_per_file.setdefault(fname, {}).update(d)
            if collect_gt:
                gt_masks = gt_dict

        t1 = time.perf_counter()
        total_inf_time = t1 - t0
        avg_time_per_image_prompt = total_inf_time / max(num_images * len(prompt_list), 1)

        write_inference_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            prompts=prompt_list,
            num_images=num_images,
            total_time=total_inf_time,
            avg_time_per_image_prompt=avg_time_per_image_prompt,
        )

        # Stacked visuals [INPUT | GT | P1 | P2 | ...] with labels on top border
        stacked_dir = os.path.join(base_pred_dir, "stacked")
        images_dir = ds.images_dir
        build_stacked_visuals(
            images_dir=images_dir,
            stacked_dir=stacked_dir,
            prompt_list=prompt_list,
            all_masks_per_file=all_masks_per_file,
            gt_masks=gt_masks,
        )

    # ------------------------------------------------------------
    # FULL MODE: train + eval + infer
    # ------------------------------------------------------------
    elif args.mode == "full":
        # 1) Train
        train_ds, val_ds = build_datasets(args)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_train = len(train_ds)
        num_val = len(val_ds)
        num_test = len(build_test_dataset(args)) if args.dataset == "cracks" else 0

        prompt_list = get_prompt_list(args.dataset)
        train_prompt = prompt_list[0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        epoch_times = []
        best_dice = 0.0

        print("\n[FULL MODE] Training...")
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            t0 = time.perf_counter()
            train_loss = train_one_epoch(
                model, train_loader, optimizer=optimizer, device=device, prompt_str=train_prompt
            )
            t1 = time.perf_counter()
            epoch_time = t1 - t0
            epoch_times.append(epoch_time)
            print(f"Train loss ({train_prompt}): {train_loss:.4f} | epoch time: {epoch_time:.3f} s")

            prompt_metrics_last_epoch = {}
            for p in prompt_list:
                val_loss, val_iou, val_dice = eval_one_epoch(
                    model, val_loader, device, prompt_str=p
                )
                print(
                    f"[Prompt: '{p}'] "
                    f"Val loss: {val_loss:.4f} | "
                    f"mIoU: {val_iou:.4f} | Dice: {val_dice:.4f}"
                )
                prompt_metrics_last_epoch[p] = {
                    "loss": val_loss,
                    "miou": val_iou,
                    "dice": val_dice,
                }
                if val_dice > best_dice:
                    best_dice = val_dice
                    ckpt_path = os.path.join(ckpt_dir, f"{args.dataset}_best.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved best checkpoint to: {ckpt_path}")

        total_train_time = sum(epoch_times)

        write_train_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            epoch_times=epoch_times,
            total_train_time=total_train_time,
            model_size_mb=model_size_mb,
        )
        write_prompt_metrics_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            prompt_metrics=prompt_metrics_last_epoch,
            split_name="val_last_epoch_full",
        )

        # 2) Reload best for final eval + infer
        ckpt_path = os.path.join(ckpt_dir, f"{args.dataset}_best.pth")
        print(f"\n[FULL MODE] Loading best checkpoint {ckpt_path} for eval+infer")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Final eval on val
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print("\n[FULL MODE] Final Eval on all prompts:")
        prompt_metrics_final = {}
        for p in prompt_list:
            val_loss, val_iou, val_dice = eval_one_epoch(
                model, val_loader, device, prompt_str=p
            )
            print(
                f"[Prompt: '{p}'] "
                f"Val loss: {val_loss:.4f} | "
                f"mIoU: {val_iou:.4f} | Dice: {val_dice:.4f}"
            )
            prompt_metrics_final[p] = {
                "loss": val_loss,
                "miou": val_iou,
                "dice": val_dice,
            }

        write_prompt_metrics_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            prompt_metrics=prompt_metrics_final,
            split_name="val_final_full",
        )

        # 3) Infer + stacked
        if args.dataset == "cracks":
            ds = build_test_dataset(args)
            base_pred_dir = os.path.join(preds_root, "cracks")
        else:
            ds = val_ds
            base_pred_dir = os.path.join(preds_root, "taping")

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_images = len(ds)
        all_masks_per_file = {}
        gt_masks = {}
        t0 = time.perf_counter()

        for idx, p in enumerate(prompt_list):
            save_dir = os.path.join(base_pred_dir, prompt_to_tag(p))
            collect_gt = idx == 0
            masks_for_p, gt_dict = run_inference(
                model,
                loader,
                device,
                save_dir,
                prompt_str=p,
                return_masks=True,
                collect_gt=collect_gt,
            )
            for fname, d in masks_for_p.items():
                all_masks_per_file.setdefault(fname, {}).update(d)
            if collect_gt:
                gt_masks = gt_dict

        t1 = time.perf_counter()
        total_inf_time = t1 - t0
        avg_time_per_image_prompt = total_inf_time / max(num_images * len(prompt_list), 1)

        write_inference_report(
            dataset=args.dataset,
            out_dir=dataset_report_dir,
            prompts=prompt_list,
            num_images=num_images,
            total_time=total_inf_time,
            avg_time_per_image_prompt=avg_time_per_image_prompt,
        )

        stacked_dir = os.path.join(base_pred_dir, "stacked")
        images_dir = ds.images_dir
        build_stacked_visuals(
            images_dir=images_dir,
            stacked_dir=stacked_dir,
            prompt_list=prompt_list,
            all_masks_per_file=all_masks_per_file,
            gt_masks=gt_masks,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
