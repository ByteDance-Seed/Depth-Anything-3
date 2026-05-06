"""Run fine-tuned DA3METRIC-LARGE on the drone_v59_depth valid split.

Loads the epoch_004 checkpoint, runs inference on every valid image (or -n
randomly-sampled images), and writes:

  <out_dir>/depth_npy/<stem>.npy      predicted metric depth at GT resolution
  <out_dir>/depth_vis/<stem>.png      colorized prediction with colorbar
  <out_dir>/comparison/<stem>.png     [RGB | GT | pred | abs-error] panel
  <out_dir>/metrics.csv               per-image AbsRel / delta<1.25 / RMSE

Predictions come out at proc_res (504). They are resized down to orig_res
(256) before saving so everything lines up with the GT depth maps.

Usage:
    python scripts/run_valid_inference.py                         # all images
    python scripts/run_valid_inference.py -n 5                    # quick test
    python scripts/run_valid_inference.py --epoch 4 --out-dir ...
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
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "drone_v59_depth"
DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints" / "drone_v59" / "run_20260429_212741_depth"
DEFAULT_OUT_DIR = REPO_ROOT / "workspace" / "valid_inference"

PRETRAINED_ID = "depth-anything/DA3METRIC-LARGE"
CANONICAL_FOCAL = 300.0
DATASET_FOCAL_AT_ORIG = 300.0
ORIG_RES = 256
PROC_RES = 504


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--num-samples", type=int, default=None,
                   help="If set, randomly sample this many images (test mode: use 5)")
    p.add_argument("--epoch", type=int, default=4,
                   help="Epoch checkpoint to load (default: 4)")
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR,
                   help="Directory containing epoch_NNN.pt files")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="valid")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Images per forward pass (default: 64)")
    p.add_argument("--save-workers", type=int, default=8,
                   help="Threads for background PNG/NPY saving (default: 8)")
    p.add_argument("--rerender-only", action="store_true",
                   help="Skip inference; rebuild PNGs from existing depth_npy/ + metrics.csv")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def list_valid_stems(split_dir: Path) -> list[str]:
    """Mirror scripts/finetune/dataset.py: require matching jpeg+npy and non-empty depth."""
    images_dir = split_dir / "images"
    depths_dir = split_dir / "depths"
    stems: list[str] = []
    for p in sorted(depths_dir.glob("*.npy")):
        stem = p.stem
        if not (images_dir / f"{stem}.jpeg").is_file():
            continue
        arr = np.load(p, mmap_mode="r")
        if np.asarray(arr).max() <= 0:
            continue
        stems.append(stem)
    return stems


def load_finetuned_model(ckpt_path: Path, device: torch.device) -> DepthAnything3:
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


# These functions are called from worker threads. They use the OO matplotlib
# API only (Figure + FigureCanvasAgg) — pyplot holds process-global state and
# is NOT thread-safe.


def save_depth_vis(depth_m: np.ndarray, out_path: Path) -> None:
    lo, hi = np.percentile(depth_m, 2), np.percentile(depth_m, 98)
    fig = Figure(figsize=(6, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(depth_m, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("depth (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def save_comparison(
    rgb: np.ndarray,
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    valid: np.ndarray,
    stem: str,
    metrics: dict,
    out_path: Path,
) -> None:
    """6-panel comparison figure.

    Top row:
      [0] RGB input
      [1] GT depth (only pixels flagged valid are drawn; others transparent)
      [2] Pred depth — colormap locked to the GT range, so you can directly
          eyeball whether the model predicts the same distance as GT
      [3] Pred depth — colormap auto-scaled to the prediction's own 2/98
          percentile, so you can see the full predicted structure
    Bottom row:
      [4] |pred - gt| on the valid mask (where metrics are computed)
      [5] Pred-vs-GT scatter, log-log, with y=x reference and ±25% band
    """
    valid_gt = valid & np.isfinite(gt_m) & (gt_m > 0)
    gt_lo = gt_hi = None
    if valid_gt.any():
        gt_lo = float(np.percentile(gt_m[valid_gt], 2))
        gt_hi = float(np.percentile(gt_m[valid_gt], 98))
        if gt_lo == gt_hi:
            # GT is a single value (e.g. one object at one distance) — widen
            # so the panel isn't pure black.
            gt_lo = gt_lo - 0.5
            gt_hi = gt_hi + 0.5
    if valid_gt.any():
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

    # GT
    if gt_lo is not None:
        gt_im = ax_gt.imshow(
            np.where(valid_gt, gt_m, np.nan),
            cmap="inferno", vmin=gt_lo, vmax=gt_hi,
        )
        ax_gt.set_title(
            f"GT depth  |  n={int(valid_gt.sum()):,} px\n"
            f"range={gt_m[valid_gt].min():.2f}–{gt_m[valid_gt].max():.2f} m",
            fontsize=13,
        )
        fig.colorbar(gt_im, ax=ax_gt, fraction=0.046, pad=0.04).set_label("depth (m)")
    else:
        ax_gt.imshow(np.zeros_like(gt_m))
        ax_gt.set_title("GT depth  |  NO VALID PIXELS", fontsize=13)

    # Pred (shared scale with GT)
    if gt_lo is not None:
        shared_im = ax_pred_shared.imshow(pred_m, cmap="inferno", vmin=gt_lo, vmax=gt_hi)
        ax_pred_shared.set_title(
            f"Pred @ GT scale ({gt_lo:.2f}–{gt_hi:.2f} m)\n"
            "direct comparison — same colorbar as GT",
            fontsize=13,
        )
        fig.colorbar(shared_im, ax=ax_pred_shared, fraction=0.046, pad=0.04).set_label("depth (m)")
    else:
        ax_pred_shared.imshow(pred_m, cmap="inferno")
        ax_pred_shared.set_title("Pred (no GT range available)", fontsize=13)

    # Pred (own 2/98 percentile)
    own_im = ax_pred_own.imshow(pred_m, cmap="inferno", vmin=pred_lo, vmax=pred_hi)
    ax_pred_own.set_title(
        f"Pred @ own scale ({pred_lo:.2f}–{pred_hi:.2f} m)\n"
        f"on valid mask  |  min={pred_min:.2f} max={pred_max:.2f} m",
        fontsize=13,
    )
    fig.colorbar(own_im, ax=ax_pred_own, fraction=0.046, pad=0.04).set_label("depth (m)")

    # Error
    e_im = ax_err.imshow(err, cmap="magma", vmin=0.0, vmax=err_hi)
    if valid_gt.any():
        err_valid = np.abs(pred_m[valid_gt] - gt_m[valid_gt])
        err_title = (
            f"|pred - gt|  (on valid mask)\n"
            f"mean={err_valid.mean():.3f}  median={np.median(err_valid):.3f} m"
        )
    else:
        err_title = "|pred - gt|  (no valid pixels)"
    ax_err.set_title(err_title, fontsize=13)
    fig.colorbar(e_im, ax=ax_err, fraction=0.046, pad=0.04).set_label("error (m)")

    for ax in (ax_rgb, ax_gt, ax_pred_shared, ax_pred_own, ax_err):
        ax.set_xticks([])
        ax.set_yticks([])

    # Scatter: pred vs gt on the valid mask
    if valid_gt.any():
        g = gt_m[valid_gt]
        p = pred_m[valid_gt]
        lo_ax = float(min(g.min(), p.min()))
        hi_ax = float(max(g.max(), p.max()))
        # Pad a little so y=x lines at the boundary are visible
        pad = 0.05 * (hi_ax - lo_ax if hi_ax > lo_ax else 1.0)
        ax_scatter.scatter(g, p, s=6, alpha=0.35, edgecolors="none")
        ax_scatter.plot([lo_ax, hi_ax], [lo_ax, hi_ax], "k--", lw=1, label="y = x")
        ax_scatter.plot([lo_ax, hi_ax], [lo_ax * 1.25, hi_ax * 1.25],
                        color="gray", linestyle=":", lw=1, label="±25%")
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
        ax_scatter.text(
            0.5, 0.5, "no valid GT pixels for scatter", ha="center", va="center",
            transform=ax_scatter.transAxes, fontsize=14,
        )
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])

    fig.suptitle(
        f"{stem}\n"
        f"AbsRel={metrics['absrel']:.3f}   "
        f"delta1={metrics['delta1']:.3f}   "
        f"RMSE={metrics['rmse']:.3f} m   "
        f"n_valid={metrics['n_valid']:,}",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def save_all_for_stem(
    stem: str,
    pred_m: np.ndarray,
    rgb_vis: np.ndarray,
    gt: np.ndarray,
    gt_valid: np.ndarray,
    metrics: dict,
    npy_dir: Path,
    vis_dir: Path,
    cmp_dir: Path,
) -> None:
    """Worker-thread entry point: persist the npy + both PNG visualizations."""
    np.save(npy_dir / f"{stem}.npy", pred_m.astype(np.float32))
    save_depth_vis(pred_m, vis_dir / f"{stem}.png")
    save_comparison(rgb_vis, gt, pred_m, gt_valid, stem, metrics, cmp_dir / f"{stem}.png")


def compute_metrics(pred_m: np.ndarray, gt_m: np.ndarray, valid: np.ndarray) -> dict:
    mask = valid & np.isfinite(gt_m) & (gt_m > 0) & np.isfinite(pred_m) & (pred_m > 0)
    if not mask.any():
        return {"absrel": float("nan"), "delta1": float("nan"), "rmse": float("nan"), "n_valid": 0}
    p = pred_m[mask]
    g = gt_m[mask]
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    return {"absrel": absrel, "delta1": delta1, "rmse": rmse, "n_valid": int(mask.sum())}


def rerender_from_disk(args: argparse.Namespace) -> None:
    """Rebuild comparison + depth_vis PNGs from saved .npy predictions.

    Assumes a previous full run wrote depth_npy/<stem>.npy. GT is reloaded
    from the original dataset. No GPU is touched.
    """
    out_root = args.out_dir
    npy_dir = out_root / "depth_npy"
    vis_dir = out_root / "depth_vis"
    cmp_dir = out_root / "comparison"
    assert npy_dir.is_dir(), f"no predictions to rerender at {npy_dir}"
    for d in (vis_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    split_dir = args.data_root / args.split
    images_dir = split_dir / "images"
    depths_dir = split_dir / "depths"

    stems = sorted(p.stem for p in npy_dir.glob("*.npy"))
    print(f"[rerender] {len(stems)} predictions under {npy_dir}")

    _ = colormaps["inferno"], colormaps["magma"]

    rows = [("stem", "absrel", "delta1", "rmse", "n_valid")]
    absrels, delta1s, rmses = [], [], []
    pbar = tqdm(total=len(stems), desc="rerender", unit="img")

    def _on_done(fut):
        exc = fut.exception()
        if exc is not None:
            raise exc
        pbar.update(1)

    with ThreadPoolExecutor(
        max_workers=args.save_workers, thread_name_prefix="save"
    ) as saver:
        for stem in stems:
            pred_m = np.load(npy_dir / f"{stem}.npy").astype(np.float32)
            gt = np.load(depths_dir / f"{stem}.npy").astype(np.float32)
            gt_valid = np.isfinite(gt) & (gt > 0)

            rgb_vis = cv2.cvtColor(
                cv2.imread(str(images_dir / f"{stem}.jpeg")), cv2.COLOR_BGR2RGB
            )
            if rgb_vis.shape[:2] != (ORIG_RES, ORIG_RES):
                rgb_vis = cv2.resize(
                    rgb_vis, (ORIG_RES, ORIG_RES), interpolation=cv2.INTER_AREA
                )

            metrics = compute_metrics(pred_m, gt, gt_valid)
            if np.isfinite(metrics["absrel"]):
                absrels.append(metrics["absrel"])
            if np.isfinite(metrics["delta1"]):
                delta1s.append(metrics["delta1"])
            if np.isfinite(metrics["rmse"]):
                rmses.append(metrics["rmse"])

            fut = saver.submit(
                save_all_for_stem,
                stem, pred_m, rgb_vis, gt, gt_valid, metrics,
                npy_dir, vis_dir, cmp_dir,
            )
            fut.add_done_callback(_on_done)

            rows.append(
                (stem, f"{metrics['absrel']:.6f}", f"{metrics['delta1']:.6f}",
                 f"{metrics['rmse']:.6f}", metrics["n_valid"])
            )
    pbar.close()

    with (out_root / "metrics.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    n = len(absrels)
    if n:
        print(
            f"\n[rerender summary over {n} images]  "
            f"AbsRel={sum(absrels)/n:.4f}  "
            f"delta1={sum(delta1s)/len(delta1s):.4f}  "
            f"RMSE={sum(rmses)/len(rmses):.4f} m"
        )
    print(f"[out] rewrote PNGs under {out_root}/")


def main() -> None:
    args = parse_args()

    if args.rerender_only:
        rerender_from_disk(args)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    split_dir = args.data_root / args.split
    images_dir = split_dir / "images"
    depths_dir = split_dir / "depths"
    assert images_dir.is_dir(), f"missing {images_dir}"
    assert depths_dir.is_dir(), f"missing {depths_dir}"

    stems = list_valid_stems(split_dir)
    print(f"[data] {len(stems)} valid images in {split_dir}")
    if args.num_samples is not None and args.num_samples < len(stems):
        rng = random.Random(args.seed)
        stems = rng.sample(stems, args.num_samples)
        print(f"[data] randomly sampled {len(stems)} images (seed={args.seed})")

    ckpt_path = args.ckpt_dir / f"epoch_{args.epoch:03d}.pt"
    assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"
    model = load_finetuned_model(ckpt_path, device)
    net = model.model
    net.eval()

    input_processor = InputProcessor()

    out_root = args.out_dir
    npy_dir = out_root / "depth_npy"
    vis_dir = out_root / "depth_vis"
    cmp_dir = out_root / "comparison"
    for d in (npy_dir, vis_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # focal at processed res: used to undo canonical_focal normalization
    focal_processed = DATASET_FOCAL_AT_ORIG * PROC_RES / ORIG_RES
    print(f"[conv] focal_processed={focal_processed:.3f}  "
          f"canonical_focal={CANONICAL_FOCAL}")

    rows = [("stem", "absrel", "delta1", "rmse", "n_valid")]
    absrels, delta1s, rmses = [], [], []

    batch_size = args.batch_size
    n_batches = (len(stems) + batch_size - 1) // batch_size

    # Warm matplotlib's colormap registry on the main thread to avoid a
    # first-call initialization race between worker threads.
    _ = colormaps["inferno"], colormaps["magma"]

    pending: list = []
    pbar = tqdm(total=len(stems), desc="infer", unit="img")

    def _on_done(fut):
        # Surface any worker exception so it doesn't get silently swallowed.
        exc = fut.exception()
        if exc is not None:
            raise exc
        pbar.update(1)

    with ThreadPoolExecutor(
        max_workers=args.save_workers, thread_name_prefix="save"
    ) as saver:
        for b in range(n_batches):
            batch_stems = stems[b * batch_size : (b + 1) * batch_size]
            batch_paths = [str(images_dir / f"{s}.jpeg") for s in batch_stems]

            # Preprocess the whole batch at once (N, 3, H, W)
            rgb_tensor, _, _ = input_processor(
                batch_paths,
                process_res=PROC_RES,
                process_res_method="upper_bound_resize",
                num_workers=1,
                sequential=True,
                desc=None,
            )
            # -> (N, 1, 3, H, W) for DA3's view dim
            x = rgb_tensor.to(device, non_blocking=True).unsqueeze(1)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = net(x)
                depth_out = out["depth"]
            # Normalize to (N, H, W): squeeze any singleton dims between N and HxW.
            while depth_out.dim() > 3:
                if depth_out.shape[1] != 1:
                    raise RuntimeError(
                        f"unexpected depth shape: {tuple(depth_out.shape)}"
                    )
                depth_out = depth_out.squeeze(1)
            preds_raw = depth_out.float().cpu().numpy()
            preds_m_proc = preds_raw * focal_processed / CANONICAL_FOCAL

            # Compute metrics + prep arrays on the main thread (cheap, numpy-
            # level), then hand off the heavy PNG rendering to the thread pool.
            for i, stem in enumerate(batch_stems):
                img_path = batch_paths[i]
                gt = np.load(depths_dir / f"{stem}.npy").astype(np.float32)
                gt_valid = np.isfinite(gt) & (gt > 0)

                pred_m = cv2.resize(
                    preds_m_proc[i], (ORIG_RES, ORIG_RES), interpolation=cv2.INTER_LINEAR
                )

                rgb_vis = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                if rgb_vis.shape[:2] != (ORIG_RES, ORIG_RES):
                    rgb_vis = cv2.resize(
                        rgb_vis, (ORIG_RES, ORIG_RES), interpolation=cv2.INTER_AREA
                    )

                metrics = compute_metrics(pred_m, gt, gt_valid)
                if np.isfinite(metrics["absrel"]):
                    absrels.append(metrics["absrel"])
                if np.isfinite(metrics["delta1"]):
                    delta1s.append(metrics["delta1"])
                if np.isfinite(metrics["rmse"]):
                    rmses.append(metrics["rmse"])

                fut = saver.submit(
                    save_all_for_stem,
                    stem, pred_m, rgb_vis, gt, gt_valid, metrics,
                    npy_dir, vis_dir, cmp_dir,
                )
                fut.add_done_callback(_on_done)
                pending.append(fut)

                rows.append(
                    (stem, f"{metrics['absrel']:.6f}", f"{metrics['delta1']:.6f}",
                     f"{metrics['rmse']:.6f}", metrics["n_valid"])
                )

        # Exiting the `with` waits for every saver task to finish.
    pbar.close()
    # Re-raise the first failure if any task errored and its callback didn't run yet.
    for fut in pending:
        exc = fut.exception()
        if exc is not None:
            raise exc

    with (out_root / "metrics.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    n = len(absrels)
    if n:
        print(
            f"\n[summary over {n} images]  "
            f"AbsRel={sum(absrels)/n:.4f}  "
            f"delta1={sum(delta1s)/len(delta1s):.4f}  "
            f"RMSE={sum(rmses)/len(rmses):.4f} m"
        )
    print(f"[out] wrote outputs under {out_root}/")


if __name__ == "__main__":
    main()
