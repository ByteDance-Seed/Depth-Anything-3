"""Run a fine-tuned DA3METRIC-LARGE checkpoint across one or more epochs.

For each epoch, writes a per-epoch subdirectory:
  <out_root>/epoch_NNN/depth_npy/<stem>.npy        predicted metric depth (m)
  <out_root>/epoch_NNN/depth_vis/<stem>.png        colorized prediction
  <out_root>/epoch_NNN/comparison/<stem>.png       RGB | pred (no-GT)
                                                   RGB | GT | pred@GT | pred@own / err / scatter (with-GT)
  <out_root>/epoch_NNN/metrics.csv                 per-image metrics (with-GT only)

Point the notebook's OUT_DIR at a specific <out_root>/epoch_NNN/ to browse.

Usage:
    # v60 valid split (has GT) → notebook-compatible metrics + 6-panel
    python scripts/run_inference.py \\
        --epochs 2 3 4 19 \\
        --ckpt-dir checkpoints/drone_v60/run_20260504_191442_depth \\
        --images-dir datasets/drone_v60_depth/valid/images \\
        --depths-dir datasets/drone_v60_depth/valid/depths \\
        --out-root workspace/v60_valid_inference

    # KBZ images (no GT) → 2-panel compare, no metrics
    python scripts/run_inference.py \\
        --epochs 2 3 4 19 \\
        --ckpt-dir checkpoints/drone_v60/run_20260504_191442_depth \\
        --images-dir datasets/KBZ_4_9/Images \\
        --out-root workspace/kbz_4_9_inference
"""

from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.io.input_processor import InputProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]

PRETRAINED_ID = "depth-anything/DA3METRIC-LARGE"
CANONICAL_FOCAL = 300.0
DATASET_FOCAL_AT_ORIG = 300.0
ORIG_RES = 256
PROC_RES = 504

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, nargs="+", required=True,
                   help="Epoch numbers to run, e.g. --epochs 2 3 4 19")
    p.add_argument("--ckpt-dir", type=Path, required=True,
                   help="Directory containing epoch_NNN.pt files")
    p.add_argument("--images-dir", type=Path, required=True,
                   help="Directory of input images (.jpg/.jpeg/.png)")
    p.add_argument("--depths-dir", type=Path, default=None,
                   help="Optional directory of GT depth .npy files. If given, "
                        "metrics + 6-panel comparison are produced.")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Output root; per-epoch subdirs are created under it")
    p.add_argument("-n", "--num-samples", type=int, default=None,
                   help="If set, randomly sample this many images")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rerender-only", action="store_true",
                   help="Skip inference; rebuild PNGs (+ metrics.csv if GT) "
                        "from existing <out_root>/epoch_NNN/depth_npy/.")
    return p.parse_args()


def list_stems(images_dir: Path, depths_dir: Path | None) -> tuple[list[str], str]:
    """Return (stems, image_ext). If depths_dir is given, require a matching
    non-empty depth .npy for every stem (mirrors finetune/dataset.py)."""
    for ext in IMAGE_EXTS:
        candidates = sorted(p for p in images_dir.glob(f"*{ext}"))
        if candidates:
            image_ext = ext
            break
    else:
        raise RuntimeError(f"no {IMAGE_EXTS} images found in {images_dir}")

    if depths_dir is None:
        return [p.stem for p in candidates], image_ext

    stems: list[str] = []
    for p in sorted(depths_dir.glob("*.npy")):
        stem = p.stem
        if not (images_dir / f"{stem}{image_ext}").is_file():
            continue
        if np.asarray(np.load(p, mmap_mode="r")).max() <= 0:
            continue
        stems.append(stem)
    return stems, image_ext


def load_finetuned_model(ckpt_path: Path, device: torch.device,
                         model: DepthAnything3 | None = None) -> DepthAnything3:
    if model is None:
        print(f"[model] loading base {PRETRAINED_ID}")
        model = DepthAnything3.from_pretrained(PRETRAINED_ID)
    print(f"[ckpt]  loading {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ckpt]  missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[ckpt]  unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.to(device=device).eval()
    return model


# Worker-thread helpers: use only the OO matplotlib API (Figure +
# FigureCanvasAgg) — pyplot holds process-global state and is NOT thread-safe.


def save_depth_vis(depth_m: np.ndarray, out_path: Path) -> None:
    lo, hi = np.percentile(depth_m, 2), np.percentile(depth_m, 98)
    fig = Figure(figsize=(6, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(depth_m, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("depth (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def save_comparison_no_gt(rgb: np.ndarray, depth_m: np.ndarray, stem: str,
                          out_path: Path) -> None:
    lo = float(np.percentile(depth_m, 2))
    hi = float(np.percentile(depth_m, 98))
    fig = Figure(figsize=(13, 6))
    FigureCanvasAgg(fig)
    ax_rgb = fig.add_subplot(1, 2, 1)
    ax_d = fig.add_subplot(1, 2, 2)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB")
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
    im = ax_d.imshow(depth_m, cmap="inferno", vmin=lo, vmax=hi)
    ax_d.set_title(f"Pred depth ({lo:.2f}-{hi:.2f} m)\n"
                   f"min={depth_m.min():.2f} max={depth_m.max():.2f}")
    ax_d.set_xticks([]); ax_d.set_yticks([])
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04).set_label("depth (m)")
    fig.suptitle(stem, fontsize=12)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def save_comparison_with_gt(rgb: np.ndarray, gt_m: np.ndarray, pred_m: np.ndarray,
                            valid: np.ndarray, stem: str, metrics: dict,
                            out_path: Path) -> None:
    """6-panel figure: RGB | GT | pred@GT scale | pred@own scale / |err| / scatter."""
    valid_gt = valid & np.isfinite(gt_m) & (gt_m > 0)
    gt_lo = gt_hi = None
    if valid_gt.any():
        gt_lo = float(np.percentile(gt_m[valid_gt], 2))
        gt_hi = float(np.percentile(gt_m[valid_gt], 98))
        if gt_lo == gt_hi:
            gt_lo -= 0.5
            gt_hi += 0.5
        pred_on_valid = pred_m[valid_gt]
        pred_lo = float(np.percentile(pred_on_valid, 2))
        pred_hi = float(np.percentile(pred_on_valid, 98))
        pred_min = float(pred_on_valid.min())
        pred_max = float(pred_on_valid.max())
    else:
        pred_lo = float(np.percentile(pred_m, 2))
        pred_hi = float(np.percentile(pred_m, 98))
        pred_min = float(pred_m.min())
        pred_max = float(pred_m.max())

    err = np.where(valid_gt, np.abs(pred_m - gt_m), np.nan)
    err_hi = float(np.nanpercentile(err, 98)) if np.isfinite(err).any() else 1.0

    fig = Figure(figsize=(24, 11))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.15)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_pred_shared = fig.add_subplot(gs[0, 2])
    ax_pred_own = fig.add_subplot(gs[0, 3])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1:3])

    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB input", fontsize=14)

    if gt_lo is not None:
        gt_im = ax_gt.imshow(np.where(valid_gt, gt_m, np.nan),
                             cmap="inferno", vmin=gt_lo, vmax=gt_hi)
        ax_gt.set_title(
            f"GT depth  |  n={int(valid_gt.sum()):,} px\n"
            f"range={gt_m[valid_gt].min():.2f}-{gt_m[valid_gt].max():.2f} m",
            fontsize=13,
        )
        fig.colorbar(gt_im, ax=ax_gt, fraction=0.046, pad=0.04).set_label("depth (m)")

        shared_im = ax_pred_shared.imshow(pred_m, cmap="inferno", vmin=gt_lo, vmax=gt_hi)
        ax_pred_shared.set_title(
            f"Pred @ GT scale ({gt_lo:.2f}-{gt_hi:.2f} m)\n"
            "direct comparison - same colorbar as GT",
            fontsize=13,
        )
        fig.colorbar(shared_im, ax=ax_pred_shared, fraction=0.046, pad=0.04).set_label("depth (m)")
    else:
        ax_gt.imshow(np.zeros_like(gt_m))
        ax_gt.set_title("GT depth  |  NO VALID PIXELS", fontsize=13)
        ax_pred_shared.imshow(pred_m, cmap="inferno")
        ax_pred_shared.set_title("Pred (no GT range available)", fontsize=13)

    own_im = ax_pred_own.imshow(pred_m, cmap="inferno", vmin=pred_lo, vmax=pred_hi)
    ax_pred_own.set_title(
        f"Pred @ own scale ({pred_lo:.2f}-{pred_hi:.2f} m)\n"
        f"on valid mask  |  min={pred_min:.2f} max={pred_max:.2f} m",
        fontsize=13,
    )
    fig.colorbar(own_im, ax=ax_pred_own, fraction=0.046, pad=0.04).set_label("depth (m)")

    e_im = ax_err.imshow(err, cmap="magma", vmin=0.0, vmax=err_hi)
    if valid_gt.any():
        err_valid = np.abs(pred_m[valid_gt] - gt_m[valid_gt])
        err_title = (f"|pred - gt|  (on valid mask)\n"
                     f"mean={err_valid.mean():.3f}  median={np.median(err_valid):.3f} m")
    else:
        err_title = "|pred - gt|  (no valid pixels)"
    ax_err.set_title(err_title, fontsize=13)
    fig.colorbar(e_im, ax=ax_err, fraction=0.046, pad=0.04).set_label("error (m)")

    for ax in (ax_rgb, ax_gt, ax_pred_shared, ax_pred_own, ax_err):
        ax.set_xticks([]); ax.set_yticks([])

    if valid_gt.any():
        g = gt_m[valid_gt]
        p = pred_m[valid_gt]
        lo_ax = float(min(g.min(), p.min()))
        hi_ax = float(max(g.max(), p.max()))
        pad = 0.05 * (hi_ax - lo_ax if hi_ax > lo_ax else 1.0)
        ax_scatter.scatter(g, p, s=6, alpha=0.35, edgecolors="none")
        ax_scatter.plot([lo_ax, hi_ax], [lo_ax, hi_ax], "k--", lw=1, label="y = x")
        ax_scatter.plot([lo_ax, hi_ax], [lo_ax * 1.25, hi_ax * 1.25],
                        color="gray", linestyle=":", lw=1, label="+/-25%")
        ax_scatter.plot([lo_ax, hi_ax], [lo_ax / 1.25, hi_ax / 1.25],
                        color="gray", linestyle=":", lw=1)
        ax_scatter.set_xlim(lo_ax - pad, hi_ax + pad)
        ax_scatter.set_ylim(lo_ax - pad, hi_ax + pad)
        ax_scatter.set_xlabel("GT depth (m)")
        ax_scatter.set_ylabel("Pred depth (m)")
        ax_scatter.set_title(
            f"Pred vs GT (valid px only)  |  "
            f"AbsRel={metrics['absrel']:.3f}  "
            f"delta1={metrics['delta1']:.3f}  "
            f"RMSE={metrics['rmse']:.3f} m",
            fontsize=13,
        )
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend(loc="upper left")
        ax_scatter.set_aspect("equal", adjustable="box")
    else:
        ax_scatter.text(0.5, 0.5, "no valid GT pixels for scatter",
                        ha="center", va="center", transform=ax_scatter.transAxes, fontsize=14)
        ax_scatter.set_xticks([]); ax_scatter.set_yticks([])

    fig.suptitle(
        f"{stem}\n"
        f"AbsRel={metrics['absrel']:.3f}   "
        f"delta1={metrics['delta1']:.3f}   "
        f"RMSE={metrics['rmse']:.3f} m   "
        f"n_valid={metrics['n_valid']:,}",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def compute_metrics(pred_m: np.ndarray, gt_m: np.ndarray, valid: np.ndarray) -> dict:
    mask = valid & np.isfinite(gt_m) & (gt_m > 0) & np.isfinite(pred_m) & (pred_m > 0)
    if not mask.any():
        return {"absrel": float("nan"), "delta1": float("nan"),
                "rmse": float("nan"), "n_valid": 0}
    p = pred_m[mask]
    g = gt_m[mask]
    ratio = np.maximum(p / g, g / p)
    return {
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "delta1": float(np.mean(ratio < 1.25)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "n_valid": int(mask.sum()),
    }


def save_all_for_stem(stem: str, pred_m: np.ndarray, rgb_vis: np.ndarray,
                      gt: np.ndarray | None, gt_valid: np.ndarray | None,
                      metrics: dict | None, npy_dir: Path, vis_dir: Path,
                      cmp_dir: Path) -> None:
    np.save(npy_dir / f"{stem}.npy", pred_m.astype(np.float32))
    save_depth_vis(pred_m, vis_dir / f"{stem}.png")
    if gt is not None:
        save_comparison_with_gt(rgb_vis, gt, pred_m, gt_valid, stem, metrics,
                                cmp_dir / f"{stem}.png")
    else:
        save_comparison_no_gt(rgb_vis, pred_m, stem, cmp_dir / f"{stem}.png")


def _load_rgb(img_path: str) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (ORIG_RES, ORIG_RES):
        rgb = cv2.resize(rgb, (ORIG_RES, ORIG_RES), interpolation=cv2.INTER_AREA)
    return rgb


def run_epoch(epoch: int, stems: list[str], image_ext: str,
              images_dir: Path, depths_dir: Path | None, out_root: Path,
              model: DepthAnything3, device: torch.device,
              input_processor: InputProcessor, batch_size: int,
              save_workers: int) -> None:
    epoch_out = out_root / f"epoch_{epoch:03d}"
    npy_dir = epoch_out / "depth_npy"
    vis_dir = epoch_out / "depth_vis"
    cmp_dir = epoch_out / "comparison"
    for d in (npy_dir, vis_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    focal_processed = DATASET_FOCAL_AT_ORIG * PROC_RES / ORIG_RES
    net = model.model
    net.eval()

    has_gt = depths_dir is not None
    rows = [("stem", "absrel", "delta1", "rmse", "n_valid")] if has_gt else None
    absrels, delta1s, rmses = [], [], []

    n_batches = (len(stems) + batch_size - 1) // batch_size
    pending: list = []
    pbar = tqdm(total=len(stems), desc=f"ep{epoch:03d}", unit="img")

    def _on_done(fut):
        exc = fut.exception()
        if exc is not None:
            raise exc
        pbar.update(1)

    with ThreadPoolExecutor(max_workers=save_workers, thread_name_prefix="save") as saver:
        for b in range(n_batches):
            batch_stems = stems[b * batch_size : (b + 1) * batch_size]
            batch_paths = [str(images_dir / f"{s}{image_ext}") for s in batch_stems]

            rgb_tensor, _, _ = input_processor(
                batch_paths, process_res=PROC_RES,
                process_res_method="upper_bound_resize",
                num_workers=1, sequential=True, desc=None,
            )
            x = rgb_tensor.to(device, non_blocking=True).unsqueeze(1)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = net(x)
                depth_out = out["depth"]
            while depth_out.dim() > 3:
                if depth_out.shape[1] != 1:
                    raise RuntimeError(f"unexpected depth shape: {tuple(depth_out.shape)}")
                depth_out = depth_out.squeeze(1)
            preds_raw = depth_out.float().cpu().numpy()
            preds_m_proc = preds_raw * focal_processed / CANONICAL_FOCAL

            for i, stem in enumerate(batch_stems):
                pred_m = cv2.resize(preds_m_proc[i], (ORIG_RES, ORIG_RES),
                                    interpolation=cv2.INTER_LINEAR)
                rgb_vis = _load_rgb(batch_paths[i])

                if has_gt:
                    gt = np.load(depths_dir / f"{stem}.npy").astype(np.float32)
                    gt_valid = np.isfinite(gt) & (gt > 0)
                    metrics = compute_metrics(pred_m, gt, gt_valid)
                    if np.isfinite(metrics["absrel"]): absrels.append(metrics["absrel"])
                    if np.isfinite(metrics["delta1"]): delta1s.append(metrics["delta1"])
                    if np.isfinite(metrics["rmse"]): rmses.append(metrics["rmse"])
                    rows.append((stem, f"{metrics['absrel']:.6f}",
                                 f"{metrics['delta1']:.6f}",
                                 f"{metrics['rmse']:.6f}", metrics["n_valid"]))
                else:
                    gt, gt_valid, metrics = None, None, None

                fut = saver.submit(save_all_for_stem, stem, pred_m, rgb_vis,
                                   gt, gt_valid, metrics, npy_dir, vis_dir, cmp_dir)
                fut.add_done_callback(_on_done)
                pending.append(fut)
    pbar.close()
    for fut in pending:
        exc = fut.exception()
        if exc is not None:
            raise exc

    if has_gt:
        with (epoch_out / "metrics.csv").open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        n = len(absrels)
        if n:
            print(f"[ep{epoch:03d}] AbsRel={sum(absrels)/n:.4f}  "
                  f"delta1={sum(delta1s)/len(delta1s):.4f}  "
                  f"RMSE={sum(rmses)/len(rmses):.4f} m  (n={n})")
    print(f"[out] epoch {epoch}: wrote {len(stems)} outputs under {epoch_out}/")


def rerender_from_disk(args: argparse.Namespace) -> None:
    """Rebuild PNGs (and metrics.csv if GT supplied) from existing depth_npy/.

    Iterates each --epochs directory under --out-root. No GPU is touched.
    """
    has_gt = args.depths_dir is not None
    for ext in IMAGE_EXTS:
        if any(args.images_dir.glob(f"*{ext}")):
            image_ext = ext
            break
    else:
        raise RuntimeError(f"no {IMAGE_EXTS} images found in {args.images_dir}")

    _ = colormaps["inferno"], colormaps["magma"]

    for epoch in args.epochs:
        epoch_out = args.out_root / f"epoch_{epoch:03d}"
        npy_dir = epoch_out / "depth_npy"
        vis_dir = epoch_out / "depth_vis"
        cmp_dir = epoch_out / "comparison"
        assert npy_dir.is_dir(), f"no predictions to rerender at {npy_dir}"
        for d in (vis_dir, cmp_dir):
            d.mkdir(parents=True, exist_ok=True)

        stems = sorted(p.stem for p in npy_dir.glob("*.npy"))
        print(f"[rerender ep{epoch:03d}] {len(stems)} predictions")

        rows = [("stem", "absrel", "delta1", "rmse", "n_valid")] if has_gt else None
        absrels, delta1s, rmses = [], [], []
        pbar = tqdm(total=len(stems), desc=f"rerender ep{epoch:03d}", unit="img")

        def _on_done(fut):
            exc = fut.exception()
            if exc is not None:
                raise exc
            pbar.update(1)

        with ThreadPoolExecutor(max_workers=args.save_workers, thread_name_prefix="save") as saver:
            for stem in stems:
                pred_m = np.load(npy_dir / f"{stem}.npy").astype(np.float32)
                rgb_vis = _load_rgb(str(args.images_dir / f"{stem}{image_ext}"))

                if has_gt:
                    gt = np.load(args.depths_dir / f"{stem}.npy").astype(np.float32)
                    gt_valid = np.isfinite(gt) & (gt > 0)
                    metrics = compute_metrics(pred_m, gt, gt_valid)
                    if np.isfinite(metrics["absrel"]): absrels.append(metrics["absrel"])
                    if np.isfinite(metrics["delta1"]): delta1s.append(metrics["delta1"])
                    if np.isfinite(metrics["rmse"]): rmses.append(metrics["rmse"])
                    rows.append((stem, f"{metrics['absrel']:.6f}",
                                 f"{metrics['delta1']:.6f}",
                                 f"{metrics['rmse']:.6f}", metrics["n_valid"]))
                else:
                    gt, gt_valid, metrics = None, None, None

                fut = saver.submit(save_all_for_stem, stem, pred_m, rgb_vis,
                                   gt, gt_valid, metrics, npy_dir, vis_dir, cmp_dir)
                fut.add_done_callback(_on_done)
        pbar.close()

        if has_gt:
            with (epoch_out / "metrics.csv").open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            n = len(absrels)
            if n:
                print(f"[ep{epoch:03d}] AbsRel={sum(absrels)/n:.4f}  "
                      f"delta1={sum(delta1s)/len(delta1s):.4f}  "
                      f"RMSE={sum(rmses)/len(rmses):.4f} m  (n={n})")
        print(f"[out] epoch {epoch}: rewrote PNGs under {epoch_out}/")


def main() -> None:
    args = parse_args()

    assert args.images_dir.is_dir(), f"missing {args.images_dir}"
    if args.depths_dir is not None:
        assert args.depths_dir.is_dir(), f"missing {args.depths_dir}"

    if args.rerender_only:
        rerender_from_disk(args)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    stems, image_ext = list_stems(args.images_dir, args.depths_dir)
    gt_tag = f" (with GT from {args.depths_dir})" if args.depths_dir else " (no GT)"
    print(f"[data] {len(stems)} {image_ext} images in {args.images_dir}{gt_tag}")

    if args.num_samples is not None and args.num_samples < len(stems):
        rng = random.Random(args.seed)
        stems = rng.sample(stems, args.num_samples)
        print(f"[data] randomly sampled {len(stems)} images (seed={args.seed})")

    _ = colormaps["inferno"], colormaps["magma"]

    print(f"[model] loading base {PRETRAINED_ID}")
    model = DepthAnything3.from_pretrained(PRETRAINED_ID)
    input_processor = InputProcessor()

    args.out_root.mkdir(parents=True, exist_ok=True)
    focal_processed = DATASET_FOCAL_AT_ORIG * PROC_RES / ORIG_RES
    print(f"[conv] focal_processed={focal_processed:.3f}  canonical_focal={CANONICAL_FOCAL}")

    for epoch in args.epochs:
        ckpt_path = args.ckpt_dir / f"epoch_{epoch:03d}.pt"
        assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"
        load_finetuned_model(ckpt_path, device, model=model)
        run_epoch(
            epoch=epoch, stems=stems, image_ext=image_ext,
            images_dir=args.images_dir, depths_dir=args.depths_dir,
            out_root=args.out_root, model=model, device=device,
            input_processor=input_processor, batch_size=args.batch_size,
            save_workers=args.save_workers,
        )

    print(f"[done] outputs under {args.out_root}/")


if __name__ == "__main__":
    main()
