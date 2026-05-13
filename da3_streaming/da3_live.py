# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
# Adapted from VGGT-Long / da3_streaming.py
# Refactored to support video/webcam/image_dir with true streaming

import argparse
import gc
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

from loop_utils.alignment_torch import apply_sim3_direct_torch, depth_to_point_cloud_optimized_torch
from loop_utils.config_utils import load_config
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)
from safetensors.torch import load_file
from depth_anything_3.api import DepthAnything3

import open3d as o3d
from pyvirtualdisplay import Display


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    is_numpy = isinstance(depth, np.ndarray)
    if is_numpy:
        depth      = torch.tensor(depth,      dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
    if device is not None:
        depth, intrinsics, extrinsics = depth.to(device), intrinsics.to(device), extrinsics.to(device)

    N, H, W = depth.shape
    dev  = depth.device
    u    = torch.arange(W, device=dev).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v    = torch.arange(H, device=dev).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=dev)
    px   = torch.cat([u, v, ones], dim=-1)

    Ki    = torch.inverse(intrinsics)
    cam   = torch.einsum("nij,nhwj->nhwi", Ki, px) * depth.unsqueeze(-1)
    cam_h = torch.cat([cam, ones], dim=-1)

    ext4 = torch.zeros(N, 4, 4, device=dev)
    ext4[:, :3, :4] = extrinsics
    ext4[:, 3, 3]   = 1.0
    c2w   = torch.inverse(ext4)
    world = torch.einsum("nij,nhwj->nhwi", c2w, cam_h)[..., :3]
    return world.cpu().numpy() if is_numpy else world


def remove_duplicates(data_list):
    seen, result = {}, []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class DA3_VideoStreaming:
    """
    --image_dir  : batch mode (carrega tudo, processa, loop closure opcional)
    --video      : streaming mode (lê chunk por chunk, memória bounded)
    --video 0    : webcam streaming (idem, lê até Ctrl+C)

    --headless   : pyvirtualdisplay + open3d, salva frames lado-a-lado → MP4
    (sem flag)   : janela OpenCV interativa
    """

    def __init__(self, video_source, save_dir, config,
                 headless=True, fps_out=5, image_dir=None,
                 frame_out_dir="./frames_result"):

        self.config        = config
        self.video_source  = video_source
        self.image_dir     = image_dir
        self.headless      = headless
        self.frame_out_dir = Path(frame_out_dir)
        self.fps_out       = fps_out
        self.frame_out_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size  = config["Model"]["chunk_size"]
        self.overlap     = config["Model"]["overlap"]
        self.overlap_s   = 0
        self.overlap_e   = self.overlap - self.overlap_s
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype       = (torch.bfloat16
                            if torch.cuda.get_device_capability()[0] >= 8
                            else torch.float16)
        self.delete_temp = config["Model"]["delete_temp_files"]
        self.loop_enable = config["Model"]["loop_enable"] and (image_dir is not None)
        if config["Model"]["loop_enable"] and image_dir is None:
            print("[AVISO] loop_closure desabilitado: requer --image_dir")

        self.output_dir           = save_dir
        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir   = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir      = os.path.join(save_dir, "_tmp_results_loop")
        self.pcd_dir              = os.path.join(save_dir, "pcd")
        for d in [self.result_unaligned_dir, self.result_aligned_dir,
                  self.result_loop_dir, self.pcd_dir]:
            os.makedirs(d, exist_ok=True)

        self.img_list              = []      # usado só no batch mode
        self.all_camera_poses      = []
        self.all_camera_intrinsics = []
        self.chunk_indices         = None
        self.sim3_list             = []
        self.loop_list             = []
        self.loop_sim3_list        = []
        self.loop_predict_list     = []
        self.loop_optimizer        = Sim3LoopOptimizer(config)
        self.result_frame_idx      = 0
        self._running_sim3         = (1.0, np.eye(3), np.zeros(3))
        self.num_chunks            = 0
        self.total_frames          = 0

        if self.headless:
            self.display = Display(visible=0, size=(1280, 720))
            self.display.start()
        else:
            self.display = None

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=not self.headless, width=640, height=480)
        self.combined_pcd = o3d.geometry.PointCloud()

        print("Loading DA3 model...")
        with open(config["Weights"]["DA3_CONFIG"]) as f:
            model_cfg = json.load(f)
        self.model = DepthAnything3(**model_cfg)
        self.model.load_state_dict(load_file(config["Weights"]["DA3"]), strict=False)
        self.model.eval().to(self.device)

        if self.loop_enable:
            lc_path = os.path.join(save_dir, "loop_closures.txt")
            self.loop_detector = LoopDetector(
                image_dir=image_dir, output=lc_path, config=config)
            self.loop_detector.load_model()

        print("Init done.")

    # ──────────────────────────────────────────────────────────────────────
    # Inferência de um chunk (aceita lista de frames diretamente)
    # ──────────────────────────────────────────────────────────────────────

    def _infer_chunk(self, frames, chunk_idx, is_loop=False, range_1=None, range_2=None):
        """Roda inferência num chunk de frames RGB e salva em disco."""
        print(f"  Inferência: {len(frames)} frames")

        ref_view = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"]

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions       = self.model.inference(frames, ref_view_strategy=ref_view)
                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0
        torch.cuda.empty_cache()

        # Salva em disco
        if is_loop:
            fname    = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
            save_dir = self.result_loop_dir
        else:
            fname    = f"chunk_{chunk_idx}.npy"
            save_dir = self.result_unaligned_dir
        np.save(os.path.join(save_dir, fname), predictions)

        return predictions

    # ──────────────────────────────────────────────────────────────────────
    # Sim3 alignment (idêntico ao upstream)
    # ──────────────────────────────────────────────────────────────────────

    def _align_2pcds(self, point_map1, conf1, point_map2, conf2,
                     chunk1_depth, chunk2_depth, chunk1_depth_conf, chunk2_depth_conf):
        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
        scale_factor   = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor, quality, method = precompute_scale_chunks_with_depth(
                chunk1_depth, chunk1_depth_conf, chunk2_depth, chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(f"  [Depth Scale] scale={scale_factor:.4f} quality={quality:.4f} method={method}")
        s, R, t = weighted_align_point_maps(
            point_map1, conf1, point_map2, conf2,
            conf_threshold=conf_threshold, config=self.config, precompute_scale=scale_factor,
        )
        print(f"  Sim3: scale={s:.4f}")
        return s, R, t

    def _align_with_previous(self, pre_pred, cur_pred):
        """Computa Sim3 entre chunk anterior e atual, atualiza running_sim3."""
        pm1 = depth_to_point_cloud_vectorized(pre_pred.depth, pre_pred.intrinsics, pre_pred.extrinsics)
        pm2 = depth_to_point_cloud_vectorized(cur_pred.depth, cur_pred.intrinsics, cur_pred.extrinsics)

        use = self.config["Model"]["align_method"] == "scale+se3"
        s, R, t = self._align_2pcds(
            pm1[-self.overlap:], pre_pred.conf[-self.overlap:],
            pm2[:self.overlap],  cur_pred.conf[:self.overlap],
            np.squeeze(pre_pred.depth[-self.overlap:]) if use else None,
            np.squeeze(cur_pred.depth[:self.overlap])  if use else None,
            np.squeeze(pre_pred.conf[-self.overlap:])  if use else None,
            np.squeeze(cur_pred.conf[:self.overlap])   if use else None,
        )
        self.sim3_list.append((s, R, t))

        # Acumula running Sim3 pra preview
        s_run, R_run, t_run = self._running_sim3
        self._running_sim3 = (s_run * s, R_run @ R, s_run * (R_run @ t) + t_run)

    def _accumulate_preview_pcd(self, predictions, chunk_idx, rep_frame_bgr):
        """Adiciona pontos ao PCD de preview (com Sim3 aplicado) e salva frame."""
        pts   = depth_to_point_cloud_vectorized(
            predictions.depth, predictions.intrinsics, predictions.extrinsics)
        conf  = predictions.conf.reshape(-1)
        thresh = np.mean(conf) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
        mask  = conf > thresh

        pts_flat = pts.reshape(-1, 3)[mask]
        if chunk_idx > 0:
            s_run, R_run, t_run = self._running_sim3
            pts_flat = s_run * (pts_flat @ R_run.T) + t_run

        chunk_pcd        = o3d.geometry.PointCloud()
        chunk_pcd.points = o3d.utility.Vector3dVector(pts_flat)
        chunk_pcd.colors = o3d.utility.Vector3dVector(
            predictions.processed_images.reshape(-1, 3)[mask] / 255.0)
        self.combined_pcd += chunk_pcd

        self._save_or_show_frame(rep_frame_bgr, chunk_idx)

    # ──────────────────────────────────────────────────────────────────────
    # Visualização
    # ──────────────────────────────────────────────────────────────────────

    def _render_pcd_screenshot(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(self.combined_pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        buf = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        return cv2.cvtColor((buf * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _save_or_show_frame(self, frame_bgr, chunk_idx):
        pcd_shot  = self._render_pcd_screenshot()
        left      = cv2.resize(frame_bgr, (640, 480))
        right     = cv2.resize(pcd_shot,  (640, 480))
        composite = np.hstack([left, right])

        if self.headless:
            out_path = self.frame_out_dir / f"frame_{self.result_frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), composite)
            print(f"  [✓] Salvo frame {self.result_frame_idx}")
        else:
            cv2.imshow("DA3 | frame | PCD", composite)
            if cv2.waitKey(1) == ord("q"):
                raise KeyboardInterrupt
        self.result_frame_idx += 1

    # ──────────────────────────────────────────────────────────────────────
    # Salva PLYs finais (aplica Sim3 acumulado aos .npy em disco)
    # ──────────────────────────────────────────────────────────────────────

    def _save_final_plys(self):
        print("\n[*] Aplicando alinhamento acumulado e salvando PLYs...")
        sim3_acc = accumulate_sim3_transforms(list(self.sim3_list))

        for chunk_idx in range(self.num_chunks - 1):
            s, R, t    = sim3_acc[chunk_idx]
            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True).item()

            world_pts = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics)
            world_pts = apply_sim3_direct_torch(world_pts, s, R, t)

            if chunk_idx == 0:
                cd0  = np.load(os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                               allow_pickle=True).item()
                pts0 = depth_to_point_cloud_vectorized(cd0.depth, cd0.intrinsics, cd0.extrinsics)
                save_confident_pointcloud_batch(
                    points=pts0, colors=cd0.processed_images, confs=cd0.conf,
                    output_path=os.path.join(self.pcd_dir, "0_pcd.ply"),
                    conf_threshold=np.mean(cd0.conf) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                    sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
                )

            flat_conf = chunk_data.conf.reshape(-1)
            save_confident_pointcloud_batch(
                points=world_pts.reshape(-1, 3),
                colors=chunk_data.processed_images.reshape(-1, 3).astype(np.uint8),
                confs=flat_conf,
                output_path=os.path.join(self.pcd_dir, f"{chunk_idx+1}_pcd.ply"),
                conf_threshold=np.mean(flat_conf) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

    # ──────────────────────────────────────────────────────────────────────
    # Camera poses
    # ──────────────────────────────────────────────────────────────────────

    def _save_camera_poses(self):
        n              = self.total_frames
        all_poses      = [None] * n
        all_intrinsics = [None] * n

        cr0, ext0 = self.all_camera_poses[0]
        _, int0   = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(cr0[0], cr0[1] - self.overlap_e)):
            w2c = np.eye(4); w2c[:3, :] = ext0[i]
            all_poses[idx]      = np.linalg.inv(w2c)
            all_intrinsics[idx] = int0[i]

        sim3_acc = accumulate_sim3_transforms(list(self.sim3_list))
        for ci in range(1, len(self.all_camera_poses)):
            cr, ext = self.all_camera_poses[ci]
            _, intr = self.all_camera_intrinsics[ci]
            s, R, t = sim3_acc[ci - 1]
            S = np.eye(4); S[:3, :3] = s * R; S[:3, 3] = t
            cr_end = cr[1] - self.overlap_e if ci < len(self.all_camera_poses) - 1 else cr[1]
            for i, idx in enumerate(range(cr[0] + self.overlap_s, cr_end)):
                if idx >= n:
                    break
                w2c = np.eye(4); w2c[:3, :] = ext[i + self.overlap_s]
                c2w = np.linalg.inv(w2c)
                tc2w = S @ c2w; tc2w[:3, :3] /= s
                all_poses[idx]      = tc2w
                all_intrinsics[idx] = intr[i + self.overlap_s]

        with open(os.path.join(self.output_dir, "camera_poses.txt"), "w") as f:
            for p in all_poses:
                if p is not None:
                    f.write(" ".join(str(x) for x in p.flatten()) + "\n")

        with open(os.path.join(self.output_dir, "intrinsic.txt"), "w") as f:
            for k in all_intrinsics:
                if k is not None:
                    f.write(f"{k[0,0]} {k[1,1]} {k[0,2]} {k[1,2]}\n")

        print(f"[✓] Poses salvas em {self.output_dir}")

    # ──────────────────────────────────────────────────────────────────────
    # Loop closure (só batch/image_dir)
    # ──────────────────────────────────────────────────────────────────────

    def _run_loop_closure(self):
        if not self.loop_enable:
            return
        print("\n[*] Detectando loop closures...")
        self.loop_list = self.loop_detector.run() or []
        self.loop_list = self.loop_detector.get_loop_list()
        del self.loop_detector
        torch.cuda.empty_cache()

        loop_results = remove_duplicates(
            process_loop_list(
                self.chunk_indices, self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
        )
        for item in loop_results:
            frames = self.img_list[item[1][0]:item[1][1]] + self.img_list[item[3][0]:item[3][1]]
            pred = self._infer_chunk(frames, None, is_loop=True, range_1=item[1], range_2=item[3])
            self.loop_predict_list.append((item, pred))

        self.loop_sim3_list = self._get_loop_sim3_from_loop_predict(self.loop_predict_list)
        input_abs      = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
        self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
        opt_abs        = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
        self._plot_loop_closure(input_abs, opt_abs)

    def _get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []
        for item in loop_predict_list:
            chunk_idx_a, chunk_a_range = item[0][0], item[0][1]
            chunk_idx_b, chunk_b_range = item[0][2], item[0][3]
            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth, item[1].intrinsics, item[1].extrinsics)
            ca_s, ca_e = 0, chunk_a_range[1] - chunk_a_range[0]
            cb_s = -(chunk_b_range[1] - chunk_b_range[0])
            cb_e = point_map_loop_org.shape[0]

            def _load(ci):
                return np.load(os.path.join(self.result_unaligned_dir, f"chunk_{ci}.npy"),
                               allow_pickle=True).item()

            def _half(ci, ls, le, rb, re):
                cd = _load(ci)
                pm = depth_to_point_cloud_vectorized(cd.depth, cd.intrinsics, cd.extrinsics)[rb:re]
                conf = cd.conf[rb:re]
                use = self.config["Model"]["align_method"] == "scale+se3"
                return self._align_2pcds(
                    pm, conf, point_map_loop_org[ls:le], item[1].conf[ls:le],
                    np.squeeze(cd.depth[rb:re]) if use else None,
                    np.squeeze(item[1].depth[ls:le]) if use else None,
                    np.squeeze(cd.conf[rb:re]) if use else None,
                    np.squeeze(item[1].conf[ls:le]) if use else None,
                )

            ca_rb = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            ca_re = ca_rb + ca_e
            cb_rb = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            cb_re = cb_rb + (chunk_b_range[1] - chunk_b_range[0])

            s_a, R_a, t_a = _half(chunk_idx_a, ca_s, ca_e, ca_rb, ca_re)
            s_b, R_b, t_b = _half(chunk_idx_b, cb_s, cb_e, cb_rb, cb_re)
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))
        return loop_sim3_list

    def _plot_loop_closure(self, before, after, fname="sim3_opt_result.png"):
        def xyz(p):
            p = p.cpu().numpy()
            return p[:, 0], p[:, 1], p[:, 2]
        x0, _, y0 = xyz(before)
        x1, _, y1 = xyz(after)
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Antes")
        plt.plot(x1, y1, "o-",  label="Depois")
        for i, j, _ in self.loop_sim3_list:
            plt.plot([x0[i], x0[j]], [y0[i], y0[j]], "r--", alpha=0.25)
            plt.plot([x1[i], x1[j]], [y1[i], y1[j]], "g-",  alpha=0.25)
        plt.gca().set_aspect("equal"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # STREAMING MODE (video / webcam)
    # Lê chunk_size frames → processa → descarta → repete
    # Memória bounded: só mantém 1 chunk + overlap em RAM
    # ══════════════════════════════════════════════════════════════════════

    def _run_streaming(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir: {self.video_source}")

        total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[*] Streaming mode | source: {self.video_source} | estimado: {total_est} frames")
        print(f"    chunk={self.chunk_size} overlap={self.overlap}")

        buffer         = []              # frames RGB do chunk atual
        chunk_idx      = 0
        pre_predictions = None
        global_frame   = 0
        step           = self.chunk_size - self.overlap

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                global_frame += 1

                # Chunk completo → processa
                if len(buffer) == self.chunk_size:
                    print(f"\n[Progress] chunk {chunk_idx+1} | frames {global_frame - self.chunk_size}..{global_frame}")

                    # Registra chunk_indices pra compatibilidade com save_camera_poses
                    start_global = global_frame - self.chunk_size
                    end_global   = global_frame
                    if self.chunk_indices is None:
                        self.chunk_indices = []
                    self.chunk_indices.append((start_global, end_global))

                    # Inferência
                    predictions = self._infer_chunk(buffer, chunk_idx)

                    # Guarda poses
                    self.all_camera_poses.append(
                        (self.chunk_indices[chunk_idx], predictions.extrinsics))
                    self.all_camera_intrinsics.append(
                        (self.chunk_indices[chunk_idx], predictions.intrinsics))

                    # Sim3 alignment
                    if chunk_idx > 0 and pre_predictions is not None:
                        self._align_with_previous(pre_predictions, predictions)

                    # Preview (PCD + frame lado-a-lado)
                    rep_bgr = cv2.cvtColor(buffer[0], cv2.COLOR_RGB2BGR)
                    self._accumulate_preview_pcd(predictions, chunk_idx, rep_bgr)

                    pre_predictions = predictions
                    chunk_idx += 1

                    # Slide window: mantém só overlap frames, descarta o resto
                    if self.overlap > 0:
                        buffer = buffer[-self.overlap:]
                    else:
                        buffer = []

            # Processa frames restantes no buffer (último chunk incompleto)
            if len(buffer) > 1:
                print(f"\n[Progress] chunk {chunk_idx+1} (final) | {len(buffer)} frames")
                start_global = global_frame - len(buffer)
                end_global   = global_frame
                if self.chunk_indices is None:
                    self.chunk_indices = []
                self.chunk_indices.append((start_global, end_global))

                predictions = self._infer_chunk(buffer, chunk_idx)
                self.all_camera_poses.append(
                    (self.chunk_indices[chunk_idx], predictions.extrinsics))
                self.all_camera_intrinsics.append(
                    (self.chunk_indices[chunk_idx], predictions.intrinsics))

                if chunk_idx > 0 and pre_predictions is not None:
                    self._align_with_previous(pre_predictions, predictions)

                rep_bgr = cv2.cvtColor(buffer[0], cv2.COLOR_RGB2BGR)
                self._accumulate_preview_pcd(predictions, chunk_idx, rep_bgr)
                chunk_idx += 1

        except KeyboardInterrupt:
            print("\n[*] Captura interrompida.")
        finally:
            cap.release()

        self.num_chunks   = chunk_idx
        self.total_frames = global_frame
        print(f"\n[✓] {global_frame} frames capturados em {chunk_idx} chunks")

    # ══════════════════════════════════════════════════════════════════════
    # BATCH MODE (image_dir)
    # Carrega tudo, processa, loop closure opcional
    # ══════════════════════════════════════════════════════════════════════

    def _run_batch(self):
        # Carrega imagens
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            paths += glob.glob(os.path.join(self.image_dir, ext))
        paths = sorted(set(paths))
        if not paths:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {self.image_dir}")
        print(f"Carregando {len(paths)} imagens de {self.image_dir}...")
        for p in paths:
            bgr = cv2.imread(p)
            if bgr is not None:
                self.img_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        print(f"[✓] {len(self.img_list)} frames prontos")

        self.total_frames = len(self.img_list)

        # Chunk indices
        n = len(self.img_list)
        if n <= self.chunk_size:
            self.chunk_indices = [(0, n)]
        else:
            step = self.chunk_size - self.overlap
            self.chunk_indices = []
            i = 0
            while True:
                s = i * step
                e = min(s + self.chunk_size, n)
                self.chunk_indices.append((s, e))
                if e == n:
                    break
                i += 1

        self.num_chunks = len(self.chunk_indices)
        print(f"[*] {n} frames | {self.num_chunks} chunks | chunk={self.chunk_size} overlap={self.overlap}")

        pre_predictions = None
        for chunk_idx in range(self.num_chunks):
            start, end = self.chunk_indices[chunk_idx]
            print(f"\n[Progress] chunk {chunk_idx+1}/{self.num_chunks}  frames ({start}, {end})")

            frames = self.img_list[start:end]
            predictions = self._infer_chunk(frames, chunk_idx)
            self.all_camera_poses.append((self.chunk_indices[chunk_idx], predictions.extrinsics))
            self.all_camera_intrinsics.append((self.chunk_indices[chunk_idx], predictions.intrinsics))
            torch.cuda.empty_cache()

            if chunk_idx > 0 and pre_predictions is not None:
                self._align_with_previous(pre_predictions, predictions)

            rep_bgr = cv2.cvtColor(self.img_list[start], cv2.COLOR_RGB2BGR)
            self._accumulate_preview_pcd(predictions, chunk_idx, rep_bgr)

            pre_predictions = predictions

        # Loop closure (só batch mode)
        self._run_loop_closure()

    # ──────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def close(self):
        self.vis.destroy_window()
        cv2.destroyAllWindows()
        if self.display is not None:
            self.display.stop()
        if self.delete_temp:
            total = 0
            for d in [self.result_unaligned_dir, self.result_aligned_dir, self.result_loop_dir]:
                for f in os.listdir(d):
                    fp = os.path.join(d, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
                        os.remove(fp)
            print(f"[✓] Temp files removidos ({total/1024**3:.2f} GiB)")

    # ──────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        if self.image_dir is not None:
            self._run_batch()        # carrega tudo + loop closure
        else:
            self._run_streaming()    # chunk por chunk, memória bounded

        # PLYs finais + poses (comum aos dois modos)
        self._save_final_plys()
        self._save_camera_poses()
        print("\n[✓] Processamento completo.")

        if self.headless and self.result_frame_idx > 0:
            out_mp4 = str(self.frame_out_dir / "output_sidebyside.mp4")
            os.system(
                f"ffmpeg -y -framerate {self.fps_out} "
                f"-i {self.frame_out_dir}/frame_%05d.jpg "
                f"-c:v libx264 -pix_fmt yuv420p {out_mp4}"
            )
            print(f"[✓] Vídeo final: {out_mp4}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA3 Streaming - video/webcam/image_dir")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video",     type=str, help="Vídeo ou webcam (ex: 0)")
    source.add_argument("--image_dir", type=str, help="Pasta de imagens jpg/png")
    parser.add_argument("--config",     type=str, default="./configs/base_config.yaml")
    parser.add_argument("--headless",   action="store_true", help="Sem janela, salva frames + MP4")
    parser.add_argument("--fps_out",    type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=None, help="Override chunk_size")
    parser.add_argument("--overlap",    type=int, default=None, help="Override overlap")
    args = parser.parse_args()

    if args.image_dir:
        video_source, image_dir = None, args.image_dir
    else:
        video_source = int(args.video) if args.video.isdigit() else args.video
        image_dir    = None

    config = load_config(args.config)
    if args.chunk_size is not None:
        config["Model"]["chunk_size"] = args.chunk_size
    if args.overlap is not None:
        config["Model"]["overlap"] = args.overlap
    if config["Model"]["overlap"] >= config["Model"]["chunk_size"]:
        parser.error(f"overlap ({config['Model']['overlap']}) >= chunk_size ({config['Model']['chunk_size']})")

    save_dir = os.path.join("./exps", f"video_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    os.makedirs(save_dir, exist_ok=True)

    if config["Model"].get("align_lib") == "numba":
        warmup_numba()

    pipeline = DA3_VideoStreaming(
        video_source=video_source, save_dir=save_dir, config=config,
        headless=args.headless, fps_out=args.fps_out,
        image_dir=image_dir, frame_out_dir="./frames_result",
    )
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[*] Interrompido.")
    finally:
        pipeline.close()

    merge_ply_files(os.path.join(save_dir, "pcd"),
                    os.path.join(save_dir, "pcd/combined_pcd.ply"))
    print(f"[✓] PCD combinado: {save_dir}/pcd/combined_pcd.ply")
    print("Concluído.")