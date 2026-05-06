"""Run fine-tuned DA3METRIC-LARGE on the KBZ_4_9 Images directory (no GT).

For each requested epoch, samples N images from the input directory,
runs inference, and writes colorized depth PNGs + raw .npy predictions
under <out_dir>/epoch_NNN/.

Usage:
    python scripts/run_kbz_inference.py --epochs 2 3 4 5 19
"""

from __future__ import annotations

import argparse
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
DEFAULT_IMAGES_DIR = REPO_ROOT / "datasets" / "KBZ_4_9" / "Images"
DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints" / "drone_v60" / "run_20260504_191442_depth"
DEFAULT_OUT_DIR = REPO_ROOT / "workspace" / "kbz_4_9_inference"

PRETRAINED_ID = "depth-anything/DA3METRIC-LARGE"
CANONICAL_FOCAL = 300.0
DATASET_FOCAL_AT_ORIG = 300.0
ORIG_RES = 256
PROC_RES = 504


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, nargs="+", required=True,
                   help="Epoch numbers to run (e.g. --epochs 2 3 4 5 19)")
    p.add_argument("-n", "--num-samples", type=int, default=400)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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


def save_side_by_side(rgb: np.ndarray, depth_m: np.ndarray, stem: str,
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


def save_all_for_stem(stem: str, pred_m: np.ndarray, rgb_vis: np.ndarray,
                      npy_dir: Path, vis_dir: Path, cmp_dir: Path) -> None:
    np.save(npy_dir / f"{stem}.npy", pred_m.astype(np.float32))
    save_depth_vis(pred_m, vis_dir / f"{stem}.png")
    save_side_by_side(rgb_vis, pred_m, stem, cmp_dir / f"{stem}.png")


def run_epoch(epoch: int, stems: list[str], images_dir: Path, out_root: Path,
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

    n_batches = (len(stems) + batch_size - 1) // batch_size
    pending: list = []
    pbar = tqdm(total=len(stems), desc=f"ep{epoch:03d}", unit="img")

    def _on_done(fut):
        exc = fut.exception()
        if exc is not None:
            raise exc
        pbar.update(1)

    with ThreadPoolExecutor(
        max_workers=save_workers, thread_name_prefix="save"
    ) as saver:
        for b in range(n_batches):
            batch_stems = stems[b * batch_size : (b + 1) * batch_size]
            batch_paths = [str(images_dir / f"{s}.jpg") for s in batch_stems]

            rgb_tensor, _, _ = input_processor(
                batch_paths,
                process_res=PROC_RES,
                process_res_method="upper_bound_resize",
                num_workers=1,
                sequential=True,
                desc=None,
            )
            x = rgb_tensor.to(device, non_blocking=True).unsqueeze(1)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = net(x)
                depth_out = out["depth"]
            while depth_out.dim() > 3:
                if depth_out.shape[1] != 1:
                    raise RuntimeError(
                        f"unexpected depth shape: {tuple(depth_out.shape)}"
                    )
                depth_out = depth_out.squeeze(1)
            preds_raw = depth_out.float().cpu().numpy()
            preds_m_proc = preds_raw * focal_processed / CANONICAL_FOCAL

            for i, stem in enumerate(batch_stems):
                img_path = batch_paths[i]
                pred_m = cv2.resize(
                    preds_m_proc[i], (ORIG_RES, ORIG_RES),
                    interpolation=cv2.INTER_LINEAR,
                )
                rgb_vis = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                if rgb_vis.shape[:2] != (ORIG_RES, ORIG_RES):
                    rgb_vis = cv2.resize(
                        rgb_vis, (ORIG_RES, ORIG_RES),
                        interpolation=cv2.INTER_AREA,
                    )

                fut = saver.submit(
                    save_all_for_stem,
                    stem, pred_m, rgb_vis,
                    npy_dir, vis_dir, cmp_dir,
                )
                fut.add_done_callback(_on_done)
                pending.append(fut)
    pbar.close()
    for fut in pending:
        exc = fut.exception()
        if exc is not None:
            raise exc
    print(f"[out] epoch {epoch}: wrote {len(stems)} outputs under {epoch_out}/")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    images_dir = args.images_dir
    assert images_dir.is_dir(), f"missing {images_dir}"

    all_stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    print(f"[data] {len(all_stems)} jpg images in {images_dir}")

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(all_stems))
    stems = rng.sample(all_stems, n)
    print(f"[data] randomly sampled {len(stems)} images (seed={args.seed})")

    # warm matplotlib colormaps
    _ = colormaps["inferno"], colormaps["magma"]

    print(f"[model] loading base {PRETRAINED_ID}")
    model = DepthAnything3.from_pretrained(PRETRAINED_ID)
    input_processor = InputProcessor()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in args.epochs:
        ckpt_path = args.ckpt_dir / f"epoch_{epoch:03d}.pt"
        assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"
        load_finetuned_model(ckpt_path, device, model=model)
        run_epoch(
            epoch=epoch,
            stems=stems,
            images_dir=images_dir,
            out_root=args.out_dir,
            model=model,
            device=device,
            input_processor=input_processor,
            batch_size=args.batch_size,
            save_workers=args.save_workers,
        )

    print(f"[done] outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
