#!/usr/bin/env python3
"""
Correct side-by-side comparison of Depth Anything V2 and V3 depth maps.

Key facts about each model's output:
  - V2 (HuggingFace Transformers): outputs **disparity** (inverse depth).
      Large values = near, small values = far. ReLU activation, arbitrary units.
  - V3 (Official repo): outputs **metric depth**.
      Small values = near, large values = far. exp() activation, metric scale.

To compare fairly we:
  1. Load raw .npy from both models (no lossy colormap round-trips).
  2. Convert V2 disparity → depth via 1/disparity.
  3. Per-frame normalize both to [0,1] using percentiles (robust to outliers).
  4. Apply the same colormap to both.
"""

import cv2
import numpy as np
import matplotlib
import argparse
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def depth_to_colormap(
    depth: np.ndarray,
    cmap: str = "Spectral",
    percentile: float = 2.0,
) -> np.ndarray:
    """Normalize a depth map (small=near, large=far) and apply a colormap.

    Convention: near → warm (red), far → cool (blue) when using Spectral.

    This mirrors V3's official ``visualize_depth`` logic:
      1. Convert depth → disparity (1/depth).
      2. Percentile-based normalization to [0,1].
      3. Flip (1-x) so that near gets the "warm" end of Spectral.
      4. Apply matplotlib colormap.
    """
    disp = np.zeros_like(depth)
    valid = depth > 0
    disp[valid] = 1.0 / depth[valid]

    if valid.sum() > 10:
        lo = np.percentile(disp[valid], percentile)
        hi = np.percentile(disp[valid], 100 - percentile)
    else:
        lo, hi = 0.0, 1.0
    if hi == lo:
        hi = lo + 1e-6

    normed = ((disp - lo) / (hi - lo)).clip(0, 1)
    normed = 1.0 - normed  # flip so near=warm

    cm = matplotlib.colormaps[cmap]
    rgb = cm(normed, bytes=False)[:, :, :3]  # H,W,3 float [0,1]
    return (rgb * 255).astype(np.uint8)       # H,W,3 uint8 RGB


def v2_disparity_to_depth(disparity: np.ndarray) -> np.ndarray:
    """Convert V2's disparity output to regular depth (same convention as V3).

    disparity: large=near, small=far  →  depth: small=near, large=far
    """
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = 1.0 / disparity[valid]
    return depth


def add_label(img: np.ndarray, text: str, bar_height: int = 36) -> np.ndarray:
    """Add a centered text label above the image (expects BGR)."""
    h, w = img.shape[:2]
    bar = np.ones((bar_height, w, 3), dtype=np.uint8) * 255
    labeled = np.vstack([bar, img])

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (bar_height + th) // 2
    cv2.putText(labeled, text, (x, y), font, scale, (0, 0, 0), thickness)
    return labeled


# ---------------------------------------------------------------------------
# Comparison pipeline
# ---------------------------------------------------------------------------

def create_comparison(
    v2_depth_dir: str,
    v3_depth_dir: str,
    input_image_dir: str,
    output_dir: str,
    cmap: str = "Spectral",
    max_frames: int | None = None,
    make_video: bool = True,
):
    """Create side-by-side comparison images from raw .npy depth files.

    Args:
        v2_depth_dir:    Directory with V2 raw depth .npy files (disparity).
        v3_depth_dir:    Directory with V3 raw depth .npy files (metric depth).
        input_image_dir: Directory with original input images.
        output_dir:      Where to write comparison JPEGs (and optional video).
        cmap:            Matplotlib colormap name (default: Spectral).
        max_frames:      Cap on number of frames to process.
        make_video:      Whether to stitch frames into an MP4 via ffmpeg.
    """
    v2_dir = Path(v2_depth_dir)
    v3_dir = Path(v3_depth_dir)
    img_dir = Path(input_image_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_files = sorted(v2_dir.glob("depth_*.npy"))
    v3_files = sorted(v3_dir.glob("depth_*.npy"))
    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.jpg"))

    n = min(len(v2_files), len(v3_files), len(img_files))
    if max_frames is not None:
        n = min(n, max_frames)

    print(f"V2 depth files : {len(v2_files)}  ({v2_dir})")
    print(f"V3 depth files : {len(v3_files)}  ({v3_dir})")
    print(f"Input images   : {len(img_files)}  ({img_dir})")
    print(f"Frames to compare: {n}")
    print(f"Colormap: {cmap}")
    print(f"Output: {out_dir}")
    print()

    for i in tqdm(range(n), desc="Comparing"):
        # --- Load raw data ---
        v2_disp = np.load(str(v2_files[i]))           # disparity
        v3_depth = np.load(str(v3_files[i]))           # metric depth
        input_bgr = cv2.imread(str(img_files[i]))

        if input_bgr is None:
            print(f"  [warn] cannot read {img_files[i]}")
            continue

        # --- Convert V2 disparity → depth ---
        v2_depth = v2_disparity_to_depth(v2_disp)

        # --- Colourmap (both using same function & convention) ---
        v2_rgb = depth_to_colormap(v2_depth, cmap=cmap)
        v3_rgb = depth_to_colormap(v3_depth, cmap=cmap)

        # --- Resize to input resolution ---
        h, w = input_bgr.shape[:2]
        v2_bgr = cv2.cvtColor(cv2.resize(v2_rgb, (w, h)), cv2.COLOR_RGB2BGR)
        v3_bgr = cv2.cvtColor(cv2.resize(v3_rgb, (w, h)), cv2.COLOR_RGB2BGR)

        # --- Assemble: Input | V2 | V3 ---
        panel = np.hstack([
            add_label(input_bgr, "Input"),
            add_label(v2_bgr, f"V2-Large ({cmap})"),
            add_label(v3_bgr, f"V3-Large ({cmap})"),
        ])

        cv2.imwrite(
            str(out_dir / f"comparison_{i:04d}.jpg"),
            panel,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

    print(f"\nWrote {n} comparison images to {out_dir}")

    # --- Optional video ---
    if make_video and n > 1:
        _stitch_video(out_dir)


def _stitch_video(out_dir: Path, fps: int = 5):
    """Stitch comparison_*.jpg into an MP4 using ffmpeg."""
    import subprocess

    video_path = out_dir / "comparison_video.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(out_dir / "comparison_*.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Video: {video_path}")
    else:
        print(f"[warn] ffmpeg failed: {result.stderr[:200]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correct side-by-side comparison of Depth Anything V2 vs V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example (using existing workspace data):
  python compare_v2_v3_depth.py \\
    --v2-depth-dir workspace/hei_chole_dav2_large/depth_raw \\
    --v3-depth-dir workspace/temp_v3_depth_check/depth_raw \\
    --input-dir    workspace/hei_chole_dav2_large/input_images \\
    --output-dir   workspace/comparison_v2_v3
""",
    )
    parser.add_argument("--v2-depth-dir", required=True,
                        help="Dir with V2 raw .npy depth files (disparity)")
    parser.add_argument("--v3-depth-dir", required=True,
                        help="Dir with V3 raw .npy depth files (metric depth)")
    parser.add_argument("--input-dir", required=True,
                        help="Dir with original input images")
    parser.add_argument("--output-dir", default="workspace/comparison_v2_v3",
                        help="Output directory (default: workspace/comparison_v2_v3)")
    parser.add_argument("--cmap", default="Spectral",
                        help="Matplotlib colormap (default: Spectral)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to compare")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video creation")
    args = parser.parse_args()

    create_comparison(
        v2_depth_dir=args.v2_depth_dir,
        v3_depth_dir=args.v3_depth_dir,
        input_image_dir=args.input_dir,
        output_dir=args.output_dir,
        cmap=args.cmap,
        max_frames=args.max_frames,
        make_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
