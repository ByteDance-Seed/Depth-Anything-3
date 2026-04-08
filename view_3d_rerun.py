#!/usr/bin/env python3
"""
Interactive 3D point cloud viewer with timeline using Rerun.

Loads per-frame depth maps, images, and camera poses from V3 inference
and displays them as a temporal 3D point cloud you can orbit and scrub.

Serves a web viewer accessible via SSH port forwarding:
  1. Run this script on the remote server
  2. SSH forward:  ssh -L 9090:localhost:9090 -L 9876:localhost:9876 <server>
  3. Open http://localhost:9090 in your browser

Usage:
    python view_3d_rerun.py \
        --data-dir workspace/3d_map_hei_chole

    # Or re-run inference on the fly:
    python view_3d_rerun.py \
        --image-dir workspace/hei_chole_dav2_large/input_images \
        --max-frames 20 --step 2
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


# ---------------------------------------------------------------------------
# Point cloud back-projection
# ---------------------------------------------------------------------------

def backproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    image: np.ndarray | None = None,
    conf: np.ndarray | None = None,
    conf_thresh: float = 0.0,
    max_points: int = 200_000,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a depth map to world-space 3D points.

    Returns (points_Nx3, colors_Nx3) or (points_Nx3, None).
    """
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if conf is not None and conf_thresh > 0:
        valid &= (conf >= conf_thresh)

    vs, us = np.where(valid)
    ds = depth[vs, us]

    # Pixel coords → camera rays
    K_inv = np.linalg.inv(K[:3, :3])
    pixels = np.stack([us, vs, np.ones_like(us)], axis=-1).astype(np.float32)  # M,3
    rays = (K_inv @ pixels.T)  # 3,M
    pts_cam = rays * ds[None, :]  # 3,M

    # Camera → world
    w2c_44 = np.eye(4, dtype=np.float32)
    w2c_44[:w2c.shape[0], :w2c.shape[1]] = w2c
    c2w = np.linalg.inv(w2c_44)
    pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])
    pts_world = (c2w @ pts_cam_h)[:3].T  # M,3

    # Colors
    colors = None
    if image is not None:
        colors = image[vs, us]  # M,3

    # Downsample if too many
    if pts_world.shape[0] > max_points:
        idx = np.random.choice(pts_world.shape[0], max_points, replace=False)
        pts_world = pts_world[idx]
        if colors is not None:
            colors = colors[idx]

    return pts_world.astype(np.float32), colors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D depth viewer with Rerun (remote web viewer)",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory from generate_3d_map.py (has exports/mini_npz/results.npz)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory with input images (alternative to --data-dir)")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-points-per-frame", type=int, default=150_000,
                        help="Max points per frame (default: 150k)")
    parser.add_argument("--conf-percentile", type=float, default=20.0,
                        help="Confidence threshold percentile (default: 20)")
    parser.add_argument("--web-port", type=int, default=9090,
                        help="Web viewer port (default: 9090)")
    parser.add_argument("--grpc-port", type=int, default=9876,
                        help="gRPC data port (default: 9876)")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save an .rrd file for later viewing")
    parser.add_argument("--serve", action="store_true",
                        help="Also start web server when --save-rrd is used")
    args = parser.parse_args()

    # ---- Load or run inference ----
    if args.data_dir:
        data_dir = Path(args.data_dir)
        npz_path = data_dir / "exports" / "mini_npz" / "results.npz"
        config_path = data_dir / "config.json"

        npz = np.load(str(npz_path))
        depth_all = npz["depth"]       # N,H,W
        conf_all = npz["conf"]         # N,H,W
        extrinsics = npz["extrinsics"] # N,3,4 or N,4,4
        intrinsics = npz["intrinsics"] # N,3,3

        # Load image paths
        with open(config_path) as f:
            config = json.load(f)
        image_paths = config["image_paths"]

        N = depth_all.shape[0]
        print(f"Loaded {N} frames from {npz_path}")

    elif args.image_dir:
        # Run V3 inference on the fly
        from depth_anything_3.api import DepthAnything3

        img_dir = Path(args.image_dir)
        image_paths = sorted(
            [str(p) for p in img_dir.glob("*.png")] +
            [str(p) for p in img_dir.glob("*.jpg")]
        )[::args.step][:args.max_frames]

        print(f"Running V3 inference on {len(image_paths)} frames...")
        model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to("cuda")
        prediction = model.inference(image=image_paths, process_res=504)

        depth_all = prediction.depth
        conf_all = prediction.conf
        extrinsics = prediction.extrinsics
        intrinsics = prediction.intrinsics
        N = depth_all.shape[0]
        print(f"Inference complete: {N} frames")
    else:
        parser.error("Provide either --data-dir or --image-dir")
        return

    # ---- Compute confidence threshold ----
    conf_thresh = np.percentile(conf_all[conf_all > 0], args.conf_percentile) if args.conf_percentile > 0 else 0.0
    print(f"Confidence threshold (p{args.conf_percentile}): {conf_thresh:.4f}")

    # ---- Load images ----
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to depth resolution
        H, W = depth_all.shape[1], depth_all.shape[2]
        img = cv2.resize(img, (W, H))
        images.append(img)

    # ---- Init Rerun ----
    rr.init("depth_anything_3d", spawn=False)

    save_only = args.save_rrd and not args.serve
    if args.save_rrd:
        rr.save(args.save_rrd)
        print(f"Will save .rrd to {args.save_rrd}")

    if not save_only:
        # Serve via web
        server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=server_uri)
        print(f"\n{'='*60}")
        print(f"Rerun web viewer serving on port {args.web_port}")
        print(f"gRPC data server on port {args.grpc_port}")
        print(f"")
        print(f"To view remotely, forward ports via SSH:")
        print(f"  ssh -L {args.web_port}:localhost:{args.web_port} "
              f"-L {args.grpc_port}:localhost:{args.grpc_port} <server>")
        print(f"Then open: http://localhost:{args.web_port}")
        print(f"{'='*60}\n")

    # ---- Log data per frame ----
    for i in range(N):
        rr.set_time("frame", sequence=i)

        depth = depth_all[i]
        conf = conf_all[i]
        K = intrinsics[i]
        ext = extrinsics[i]
        img = images[i]

        # Back-project to world points
        pts, cols = backproject_depth(
            depth, K, ext, img, conf,
            conf_thresh=conf_thresh,
            max_points=args.max_points_per_frame,
        )

        # Log point cloud
        if pts.shape[0] > 0:
            rr.log(
                "world/points",
                rr.Points3D(positions=pts, colors=cols, radii=0.002),
            )

        # Log the camera (pinhole + transform)
        w2c_44 = np.eye(4, dtype=np.float32)
        w2c_44[:ext.shape[0], :ext.shape[1]] = ext
        c2w = np.linalg.inv(w2c_44)

        rr.log(
            "world/camera",
            rr.Transform3D(mat3x3=c2w[:3, :3], translation=c2w[:3, 3]),
        )
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=K,
                width=depth.shape[1],
                height=depth.shape[0],
            ),
        )
        rr.log("world/camera/image", rr.Image(img))

        # Log depth as a separate image
        from depth_anything_3.utils.visualize import visualize_depth
        depth_vis = visualize_depth(depth)
        rr.log("world/camera/depth_vis", rr.Image(depth_vis))

        print(f"  Frame {i:3d}: {pts.shape[0]:,} points")

    print(f"\nLogged {N} frames.")

    if save_only:
        # Flush and exit cleanly
        print(f"Saved .rrd to {args.save_rrd}")
        return

    print("Viewer is running — press Ctrl+C to stop.")

    # Keep process alive
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
