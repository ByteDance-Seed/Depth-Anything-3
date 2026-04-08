#!/usr/bin/env python3
"""
Render per-frame point cloud views from multiple viewpoints.

For each frame, back-projects that single frame's depth map into a 3D point
cloud and renders it from several canonical viewpoints using Open3D's
offscreen renderer.  Each frame produces one frame_NNNN_views.png grid image.

Usage:
    # One views.png per frame
    python visualize_pointcloud.py \
        --npz workspace/3d_map_hei_chole/exports/mini_npz/results.npz \
        --images workspace/hei_chole_dav2_large/input_images \
        --output workspace/pc_vis

    # Limit to first 5 frames, custom viewpoints
    python visualize_pointcloud.py \
        --npz workspace/3d_map_hei_chole/exports/mini_npz/results.npz \
        --images workspace/hei_chole_dav2_large/input_images \
        --output workspace/pc_vis \
        --max-frames 5 \
        --viewpoints front bird top right
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


# ---------------------------------------------------------------------------
# Back-projection (single frame)
# ---------------------------------------------------------------------------

def backproject_single_frame(
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
    image: np.ndarray | None = None,
    conf: np.ndarray | None = None,
    conf_thresh: float = 1.0,
    max_points: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one depth map to world-frame points with colors.

    Returns (points, colors) both (M, 3), colors in [0, 1].
    """
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        valid &= conf >= conf_thresh

    flat_d = depth.reshape(-1)
    vidx = np.flatnonzero(valid.reshape(-1))
    if vidx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float64)

    if vidx.size > max_points:
        vidx = np.random.choice(vidx, max_points, replace=False)

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)

    K_inv = np.linalg.inv(K)
    ext44 = np.eye(4, dtype=np.float32)
    ext44[:extrinsic.shape[0], :extrinsic.shape[1]] = extrinsic
    c2w = np.linalg.inv(ext44)

    rays = (K_inv @ pix[vidx].T)         # (3, M)
    Xc = rays * flat_d[vidx][None, :]
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]), dtype=np.float32)])
    pts = (c2w @ Xc_h)[:3].T             # (M, 3)

    if image is not None:
        cols = image.reshape(-1, 3)[vidx].astype(np.float64) / 255.0
    else:
        d_vals = flat_d[vidx]
        d_norm = (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min() + 1e-8)
        cols = plt.cm.viridis(d_norm)[:, :3]

    return pts, cols


# ---------------------------------------------------------------------------
# Open3D helpers
# ---------------------------------------------------------------------------

def make_o3d_pointcloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def make_single_frustum(
    extrinsic: np.ndarray,
    K: np.ndarray,
    depth_shape: tuple[int, int],
    scale: float = 0.02,
    color: tuple = (1.0, 0.2, 0.2),
) -> o3d.geometry.LineSet:
    """Create a single camera frustum wireframe."""
    H, W = depth_shape
    ext44 = np.eye(4, dtype=np.float64)
    ext44[:extrinsic.shape[0], :extrinsic.shape[1]] = extrinsic
    c2w = np.linalg.inv(ext44)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    corners_cam = np.array([
        [(0 - cx) / fx, (0 - cy) / fy, 1.0],
        [(W - cx) / fx, (0 - cy) / fy, 1.0],
        [(W - cx) / fx, (H - cy) / fy, 1.0],
        [(0 - cx) / fx, (H - cy) / fy, 1.0],
    ]) * scale

    pts_cam = np.vstack([np.zeros(3), corners_cam])
    pts_h = np.hstack([pts_cam, np.ones((5, 1))])
    pts_world = (c2w @ pts_h.T)[:3].T

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


# ---------------------------------------------------------------------------
# Viewpoint definitions
# ---------------------------------------------------------------------------

VIEWPOINTS = {
    "front": {
        "label": "Front",
        "lookat": [0, 0, 0],
        "eye": [0, 0, -1.5],
        "up": [0, -1, 0],
    },
    "top": {
        "label": "Top-Down",
        "lookat": [0, 0, 0],
        "eye": [0, -1.5, 0],
        "up": [0, 0, 1],
    },
    "right": {
        "label": "Right Side",
        "lookat": [0, 0, 0],
        "eye": [1.5, 0, 0],
        "up": [0, -1, 0],
    },
    "left": {
        "label": "Left Side",
        "lookat": [0, 0, 0],
        "eye": [-1.5, 0, 0],
        "up": [0, -1, 0],
    },
    "bird": {
        "label": "Bird's Eye (45\u00b0)",
        "lookat": [0, 0, 0],
        "eye": [0.8, -1.2, -0.8],
        "up": [0, -1, 0],
    },
    "back": {
        "label": "Back",
        "lookat": [0, 0, 0],
        "eye": [0, 0, 1.5],
        "up": [0, -1, 0],
    },
    "bird_left": {
        "label": "Bird's Eye (Left)",
        "lookat": [0, 0, 0],
        "eye": [-0.8, -1.2, -0.8],
        "up": [0, -1, 0],
    },
}


def compute_viewpoint_params(
    vp_def: dict,
    center: np.ndarray,
    scene_scale: float,
    extrinsic: np.ndarray | None = None,
) -> dict:
    """Compute actual eye/lookat/up for one viewpoint."""
    if vp_def.get("from_camera") and extrinsic is not None:
        ext44 = np.eye(4, dtype=np.float64)
        ext44[:extrinsic.shape[0], :extrinsic.shape[1]] = extrinsic
        c2w = np.linalg.inv(ext44)
        eye = c2w[:3, 3]
        forward = c2w[:3, 2]
        up = -c2w[:3, 1]
        lookat = eye + forward * scene_scale * 0.5
        return {"eye": eye, "lookat": lookat, "up": up, "label": vp_def.get("label", "Camera")}

    eye = center + np.array(vp_def["eye"], dtype=np.float64) * scene_scale
    lookat = center + np.array(vp_def["lookat"], dtype=np.float64) * scene_scale * 0.1
    up = np.array(vp_def["up"], dtype=np.float64)
    return {"eye": eye, "lookat": lookat, "up": up, "label": vp_def.get("label", "")}


# ---------------------------------------------------------------------------
# Offscreen rendering
# ---------------------------------------------------------------------------

def render_views(
    pcd: o3d.geometry.PointCloud,
    frustum: o3d.geometry.LineSet | None,
    viewpoints: list[dict],
    width: int = 800,
    height: int = 600,
    point_size: float = 2.0,
    bg_color: tuple = (1.0, 1.0, 1.0),
) -> list[np.ndarray]:
    """Render point cloud from multiple viewpoints."""
    renderer = rendering.OffscreenRenderer(width, height)
    mat = rendering.MaterialRecord()
    mat.point_size = point_size
    mat.shader = "defaultUnlit"

    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2.0

    images = []
    for vp in viewpoints:
        renderer.scene.clear_geometry()
        renderer.scene.set_background(np.array([*bg_color, 1.0], dtype=np.float32))
        renderer.scene.add_geometry("pointcloud", pcd, mat)
        if frustum is not None:
            renderer.scene.add_geometry("camera", frustum, line_mat)

        renderer.setup_camera(
            60.0,
            np.array(vp["lookat"], dtype=np.float64),
            np.array(vp["eye"], dtype=np.float64),
            np.array(vp["up"], dtype=np.float64),
        )

        img = np.asarray(renderer.render_to_image())
        images.append(img)

    del renderer
    return images


# ---------------------------------------------------------------------------
# Matplotlib grid
# ---------------------------------------------------------------------------

def make_frame_grid(
    views: list[dict],
    frame_idx: int,
    source_image: np.ndarray | None,
    output_path: str,
    dpi: int = 150,
):
    """Create a multi-viewpoint grid for one frame.

    Layout: source image (if available) + rendered viewpoints.
    """
    has_src = source_image is not None
    n_panels = len(views) + (1 if has_src else 0)
    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle(f"Frame {frame_idx:04d}", fontsize=14, fontweight="bold")

    panel_idx = 0

    if has_src:
        r, c = divmod(panel_idx, cols)
        ax = axes[r, c]
        ax.imshow(source_image)
        ax.set_title("Input Image", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        panel_idx += 1

    for view in views:
        r, c = divmod(panel_idx, cols)
        ax = axes[r, c]
        ax.imshow(view["image"])
        ax.set_title(view["viewpoint_label"], fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        panel_idx += 1

    for idx in range(panel_idx, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render per-frame point cloud from multiple viewpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--npz", required=True,
                        help="Path to results.npz")
    parser.add_argument("--images", default=None,
                        help="Path to input image directory")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (for image path mapping)")
    parser.add_argument("--output", type=str, default="workspace/pc_vis",
                        help="Output directory")
    parser.add_argument("--viewpoints", nargs="*",
                        default=["front", "bird", "top", "right", "bird_left"],
                        help=f"Viewpoint names: {list(VIEWPOINTS.keys())}")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to render")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame step (e.g. 2 = every other frame)")
    parser.add_argument("--max-points", type=int, default=100_000,
                        help="Max points per frame (default: 100000)")
    parser.add_argument("--conf-percentile", type=float, default=20.0,
                        help="Confidence threshold percentile (default: 20)")
    parser.add_argument("--width", type=int, default=960,
                        help="Render width (default: 960)")
    parser.add_argument("--height", type=int, default=720,
                        help="Render height (default: 720)")
    parser.add_argument("--point-size", type=float, default=2.5,
                        help="Point size (default: 2.5)")
    parser.add_argument("--bg", type=str, default="white",
                        choices=["white", "black", "gray"],
                        help="Background color")
    parser.add_argument("--no-frustum", action="store_true",
                        help="Hide camera frustum")
    parser.add_argument("--no-source-image", action="store_true",
                        help="Don't include source image in grid")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    args = parser.parse_args()

    bg_colors = {"white": (1, 1, 1), "black": (0, 0, 0), "gray": (0.2, 0.2, 0.2)}
    bg = bg_colors[args.bg]

    for vp_name in args.viewpoints:
        if vp_name not in VIEWPOINTS:
            print(f"Unknown viewpoint: {vp_name}. Available: {list(VIEWPOINTS.keys())}")
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load NPZ ----
    print(f"Loading {args.npz} ...")
    npz = np.load(args.npz)
    depth_all = npz["depth"]           # (N, H, W)
    conf_all = npz.get("conf")         # (N, H, W) or None
    extrinsics = npz["extrinsics"]     # (N, 3, 4)
    intrinsics = npz["intrinsics"]     # (N, 3, 3)

    N = depth_all.shape[0]
    H, W = depth_all.shape[1], depth_all.shape[2]

    indices = list(range(0, N, args.step))
    if args.max_frames is not None:
        indices = indices[:args.max_frames]

    # Confidence threshold (global)
    conf_thresh = -np.inf
    if conf_all is not None:
        conf_thresh = float(np.percentile(conf_all, args.conf_percentile))
        print(f"Confidence threshold (p{args.conf_percentile}): {conf_thresh:.4f}")

    # ---- Load images ----
    source_images: dict[int, np.ndarray] = {}
    if args.images:
        img_dir = Path(args.images)
        image_paths = None
        if args.config and Path(args.config).exists():
            with open(args.config) as f:
                cfg = json.load(f)
                if "image_paths" in cfg:
                    image_paths = cfg["image_paths"]

        if image_paths is None:
            all_imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
            image_paths = [str(p) for p in all_imgs]

        for i in indices:
            if i < len(image_paths):
                img = cv2.imread(str(image_paths[i]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (W, H))
                    source_images[i] = img

    print(f"Rendering {len(indices)} frames x {len(args.viewpoints)} viewpoints ...")

    # ---- Per-frame loop ----
    for count, fi in enumerate(indices):
        depth = depth_all[fi]
        conf = conf_all[fi] if conf_all is not None else None
        K = intrinsics[fi]
        ext = extrinsics[fi]
        src_img = source_images.get(fi)

        pts, cols = backproject_single_frame(
            depth, K, ext,
            image=src_img, conf=conf,
            conf_thresh=conf_thresh,
            max_points=args.max_points,
        )

        if pts.shape[0] == 0:
            print(f"  Frame {fi:4d}: no valid points, skipping")
            continue

        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        center = (lo + hi) / 2
        scene_scale = max(np.linalg.norm(hi - lo), 1e-6)

        pcd = make_o3d_pointcloud(pts, cols)
        frustum = None
        if not args.no_frustum:
            frustum = make_single_frustum(
                ext, K, (H, W), scale=scene_scale * 0.03, color=(1.0, 0.2, 0.2),
            )

        viewpoints = []
        for vp_name in args.viewpoints:
            vp = compute_viewpoint_params(
                VIEWPOINTS[vp_name], center, scene_scale, extrinsic=ext,
            )
            viewpoints.append(vp)

        rendered = render_views(
            pcd, frustum, viewpoints,
            width=args.width, height=args.height,
            point_size=args.point_size, bg_color=bg,
        )

        views = [{"image": img, "viewpoint_label": vp["label"]}
                 for img, vp in zip(rendered, viewpoints)]

        grid_path = output_dir / f"frame_{fi:04d}_views.png"
        make_frame_grid(
            views, fi,
            source_image=src_img if not args.no_source_image else None,
            output_path=str(grid_path),
            dpi=args.dpi,
        )

        print(f"  Frame {fi:4d}: {pts.shape[0]:,} pts -> {grid_path.name}")

    print(f"\nDone. {len(indices)} frames saved to: {output_dir}")


if __name__ == "__main__":
    main()
