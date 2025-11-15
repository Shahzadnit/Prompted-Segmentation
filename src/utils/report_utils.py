import os
from typing import Dict, List


def _write_header(f, title: str):
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n\n")


def write_train_report(
    dataset: str,
    out_dir: str,
    num_train: int,
    num_val: int,
    num_test: int,
    epoch_times: List[float],
    total_train_time: float,
    model_size_mb: float,
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report_train.txt")

    with open(path, "w") as f:
        _write_header(f, f"Training Report - Dataset: {dataset}")
        f.write(f"Train images: {num_train}\n")
        f.write(f"Val images:   {num_val}\n")
        f.write(f"Test images:  {num_test}\n\n")

        f.write("Per-epoch training times (seconds):\n")
        for i, t in enumerate(epoch_times, start=1):
            f.write(f"  Epoch {i}: {t:.3f} s\n")
        f.write(f"\nTotal training time: {total_train_time:.3f} s\n\n")

        f.write(f"Approx. model size (trainable params): {model_size_mb:.2f} MB\n\n")

        f.write("Brief failure notes (qualitative, to refine manually if needed):\n")
        f.write("- Lower performance is often observed on:\n")
        f.write("  * very thin or hairline cracks / taping areas\n")
        f.write("  * regions near image borders or with heavy blur\n")
        f.write("  * scenes with strong lighting changes or texture clutter\n")
        f.write("Inspect stacked visualizations for concrete failure examples.\n")


def write_prompt_metrics_report(
    dataset: str,
    out_dir: str,
    prompt_metrics: Dict[str, Dict[str, float]],
    split_name: str,
):
    """
    prompt_metrics: {prompt: {'loss': float, 'miou': float, 'dice': float}}
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prompt_metrics.txt")

    with open(path, "a") as f:
        _write_header(f, f"Per-prompt Metrics - Dataset: {dataset} - Split: {split_name}")
        f.write(f"{'Prompt':40s} | {'Loss':>8s} | {'mIoU':>8s} | {'Dice':>8s}\n")
        f.write("-" * 80 + "\n")
        for p, m in prompt_metrics.items():
            f.write(
                f"{p:40s} | {m['loss']:.4f} | {m['miou']:.4f} | {m['dice']:.4f}\n"
            )


def write_inference_report(
    dataset: str,
    out_dir: str,
    prompts: List[str],
    num_images: int,
    total_time: float,
    avg_time_per_image_prompt: float,
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "inference_timings.txt")

    with open(path, "w") as f:
        _write_header(f, f"Inference Timing Report - Dataset: {dataset}")
        f.write(f"Number of images: {num_images}\n")
        f.write(f"Number of prompts: {len(prompts)}\n")
        f.write(f"Total inference time (all prompts): {total_time:.3f} s\n")
        f.write(
            f"Average time per image per prompt: {avg_time_per_image_prompt:.6f} s\n\n"
        )
        f.write("Prompts used:\n")
        for p in prompts:
            f.write(f"- {p}\n")
        f.write(
            "\nNote: Stacked visualizations are saved under predictions/<dataset>/stacked/\n"
        )
