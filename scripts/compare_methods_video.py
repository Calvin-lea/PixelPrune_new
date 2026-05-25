"""
Compare pred_2d (old) vs ad_vla (new) token selection on video frames.
Outputs side-by-side visualization with green=kept, red=dropped overlay.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixelprune.methods import get_selector


def extract_frames(video_path: str, num_frames: int = 8) -> List[np.ndarray]:
    """Extract evenly-spaced frames from video as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def image_to_patches(
    img: np.ndarray, patch_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert image to Qwen3-VL-style pixel_values and grid_thw.

    Mimics Qwen3-VL image processor: resizes to fit constraints,
    extracts 16x16 patches, normalizes each to [0,1].

    Returns:
        pixel_values: (N, 3*patch_size^2) float tensor, patch pixels in [0,1]
        grid_thw: (1, 3) tensor [temporal=1, h_patches, w_patches]
    """
    h, w = img.shape[:2]

    # Constrain to reasonable dimensions (multiple of patch_size * merge_size)
    merge_size = 2
    align = patch_size * merge_size  # 32
    max_pixels = 250880  # ~640*392 for compact visualization
    scale = min(1.0, (max_pixels / (h * w)) ** 0.5)
    new_h = int(h * scale) // align * align
    new_w = int(w * scale) // align * align
    new_h = max(align, new_h)
    new_w = max(align, new_w)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_f = img_resized.astype(np.float32) / 255.0  # [0, 1]

    h_patches = new_h // patch_size
    w_patches = new_w // patch_size

    # Extract patches: reshape to (h_patches, patch_size, w_patches, patch_size, 3)
    patches = img_f.reshape(h_patches, patch_size, w_patches, patch_size, 3)
    patches = patches.transpose(0, 2, 3, 1, 4)  # (h, w, ph, pw, 3)
    patches = patches.reshape(h_patches * w_patches, -1)  # (N, 3*ph*pw)

    pixel_values = torch.from_numpy(patches).float()
    grid_thw = torch.tensor([[1, h_patches, w_patches]])

    return pixel_values, grid_thw, img_resized


def merged_keep_to_patch_mask(
    merged_indices: torch.Tensor,
    merged_h: int,
    merged_w: int,
    patch_h: int,
    patch_w: int,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """Convert merged keep indices to a boolean mask over patch grid.

    Returns mask of shape (patch_h, patch_w), True = kept.
    """
    keep_patches = set()
    for midx in merged_indices.tolist():
        mr = midx // merged_w
        mc = midx % merged_w
        # Each merged token covers spatial_merge_size x spatial_merge_size patches
        for dr in range(spatial_merge_size):
            for dc in range(spatial_merge_size):
                pr = mr * spatial_merge_size + dr
                pc = mc * spatial_merge_size + dc
                if pr < patch_h and pc < patch_w:
                    keep_patches.add((pr, pc))

    mask = np.zeros((patch_h, patch_w), dtype=bool)
    for r, c in keep_patches:
        mask[r, c] = True
    return mask


def create_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 16,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay green (kept) / red (dropped) semi-transparent patches on image.

    Args:
        img: RGB image (H, W, 3) uint8
        mask: (patch_h, patch_w) bool, True=kept
        patch_size: pixel size of each patch
        alpha: overlay transparency
    """
    result = img.copy().astype(np.float32)
    patch_h, patch_w = mask.shape

    green = np.array([0, 200, 80], dtype=np.float32)
    red = np.array([220, 50, 50], dtype=np.float32)

    for r in range(patch_h):
        for c in range(patch_w):
            y0 = r * patch_size
            x0 = c * patch_size
            y1 = min(y0 + patch_size, img.shape[0])
            x1 = min(x0 + patch_size, img.shape[1])
            color = green if mask[r, c] else red
            result[y0:y1, x0:x1] = (1 - alpha) * result[y0:y1, x0:x1] + alpha * color

    return result.clip(0, 255).astype(np.uint8)


def draw_metrics(
    img: np.ndarray,
    method_name: str,
    num_kept: int,
    num_total: int,
    ratio: float,
    frame_idx: int,
    is_video: bool = False,
) -> np.ndarray:
    """Draw metrics text in top-left corner."""
    result = img.copy()
    h, w = result.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 900)
    thickness = max(1, int(w / 500))
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    lines = [
        f"{method_name}",
        f"Frame: {frame_idx}",
        f"Kept: {num_kept}/{num_total} ({ratio*100:.1f}%)",
    ]
    if is_video:
        lines.append("(video seq)")

    y = 30
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(result, (8, y - th - 6), (8 + tw + 12, y + 6), bg_color, -1)
        cv2.putText(result, line, (14, y), font, font_scale, color, thickness)
        y += th + 14
    return result


def process_frame(
    img: np.ndarray,
    frame_idx: int,
    sel_old,
    sel_new,
) -> dict:
    """Run both selectors on a single frame and collect results."""
    pixel_values, grid_thw, img_resized = image_to_patches(img)
    spatial_merge_size = 2
    patch_h, patch_w = int(grid_thw[0, 1]), int(grid_thw[0, 2])
    merged_h, merged_w = patch_h // spatial_merge_size, patch_w // spatial_merge_size

    # Run pred_2d (old)
    idx_old = sel_old.select(pixel_values.clone(), grid_thw, spatial_merge_size)
    num_merged = merged_h * merged_w
    num_kept_old = len(idx_old[0])
    mask_old = merged_keep_to_patch_mask(
        idx_old[0], merged_h, merged_w, patch_h, patch_w
    )

    # Run ad_vla (new)
    idx_new = sel_new.select(pixel_values.clone(), grid_thw, spatial_merge_size)
    num_kept_new = len(idx_new[0])
    mask_new = merged_keep_to_patch_mask(
        idx_new[0], merged_h, merged_w, patch_h, patch_w
    )

    # Ratio info
    num_patches = patch_h * patch_w

    return {
        "img_resized": img_resized,
        "mask_old": mask_old,
        "mask_new": mask_new,
        "num_kept_old": num_kept_old,
        "num_kept_new": num_kept_new,
        "num_merged": num_merged,
        "num_patches": num_patches,
        "ratio_old": num_kept_old / num_merged,
        "ratio_new": num_kept_new / num_merged,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "merged_h": merged_h,
        "merged_w": merged_w,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PixelPrune methods on video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "-n", "--num-frames", type=int, default=6,
        help="Number of frames to extract (default: 6)"
    )
    parser.add_argument(
        "-o", "--output", default="comparison.png",
        help="Output image path (default: comparison.png)"
    )
    args = parser.parse_args()

    print(f"Extracting {args.num_frames} frames from {args.video}...")
    frames = extract_frames(args.video, args.num_frames)
    print(f"  Got {len(frames)} frames")

    # Create selectors
    # pred_2d with mae + threshold for reasonable compression on natural images
    sel_old = get_selector("pred_2d", method="mae", threshold=0.02)
    sel_new = get_selector("ad_vla", method="mae")

    # Process all frames
    all_results = []
    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}...")
        result = process_frame(frame, i, sel_old, sel_new)
        all_results.append(result)

    # Build side-by-side grid: N rows, 2 columns
    n = len(all_results)
    rows_img = []
    for i, r in enumerate(all_results):
        img = r["img_resized"]

        overlay_old = create_overlay(img, r["mask_old"])
        overlay_old = draw_metrics(
            overlay_old, "pred_2d (old)",
            r["num_kept_old"], r["num_merged"], r["ratio_old"], i,
            is_video=(i > 0),
        )

        overlay_new = create_overlay(img, r["mask_new"])
        overlay_new = draw_metrics(
            overlay_new, "ad_vla (new)",
            r["num_kept_new"], r["num_merged"], r["ratio_new"], i,
            is_video=(i > 0),
        )

        # Concatenate old | new
        pair = np.hstack([overlay_old, overlay_new])
        rows_img.append(pair)

    # Concatenate all rows
    full_grid = np.vstack(rows_img)

    # Add column headers
    col_w = full_grid.shape[1] // 2
    header_h = 40
    header = np.zeros((header_h, full_grid.shape[1], 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, col_w / 700)
    thick = max(1, int(col_w / 400))
    for text, x_center in [("pred_2d (Old Method)", col_w // 2),
                            ("ad_vla (New Method)", col_w + col_w // 2)]:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(header, text,
                    (x_center - tw // 2, header_h // 2 + th // 2),
                    font, scale, (220, 220, 220), thick)

    full_grid = np.vstack([header, full_grid])

    cv2.imwrite(args.output, cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
    print(f"\nSaved comparison to {args.output}")

    # Print summary stats
    print("\n=== Summary ===")
    print(f"{'Frame':>6} | {'pred_2d kept':>14} | {'ad_vla kept':>14} | {'Old ratio':>10} | {'New ratio':>10}")
    print("-" * 65)
    for i, r in enumerate(all_results):
        print(f"{i:>6} | {r['num_kept_old']:>6}/{r['num_merged']:<6} | "
              f"{r['num_kept_new']:>6}/{r['num_merged']:<6} | "
              f"{r['ratio_old']:>9.1%} | {r['ratio_new']:>9.1%}")


if __name__ == "__main__":
    main()
