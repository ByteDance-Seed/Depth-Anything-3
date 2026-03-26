import glob, os, torch
from depth_anything_3.api import DepthAnything3
import imageio
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# 設定
video_path = "../datasets/Hei-Chole1_dissection.mp4"
output_dir = "workspace/hei_chole_output_new"
model_dir = "depth-anything/DA3-SMALL"
fps_extract = 5  # Extract 5 frames per second
process_res = 252
export_format = "glb-depth_vis"

# 出力ディレクトリ作成
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "input_images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "depth_vis"), exist_ok=True)

print("Loading video...")
# ビデオから画像を抽出
reader = imageio.get_reader(video_path)
video_fps = reader.get_meta_data()['fps']
print(f"Video FPS: {video_fps}")

# フレームを抽出
frame_interval = int(video_fps / fps_extract)
frames = []
frame_count = 0
extracted_count = 0

print(f"Extracting frames (every {frame_interval} frames)...")
for i, frame in enumerate(tqdm(reader)):
    if i % frame_interval == 0:
        # フレームを保存
        image_path = os.path.join(output_dir, "input_images", f"frame_{extracted_count:04d}.png")
        imageio.imwrite(image_path, frame)
        frames.append(image_path)
        extracted_count += 1

reader.close()
print(f"Extracted {len(frames)} frames")

# モデルをロード
print(f"Loading model: {model_dir}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained(model_dir)
model = model.to(device=device)

print(f"Processing {len(frames)} frames...")
# 画像を一括処理
prediction = model.inference(
    frames,
    process_res=process_res,
)

print(f"Depth shape: {prediction.depth.shape}")
print(f"Images shape: {prediction.processed_images.shape}")

# Depth visualizationを保存
print("Saving depth visualizations...")
for i in tqdm(range(len(prediction.depth))):
    depth = prediction.depth[i]
    # Normalize depth to 0-255
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    # カラーマップ適用
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    depth_path = os.path.join(output_dir, "depth_vis", f"depth_{i:04d}.png")
    cv2.imwrite(depth_path, depth_colored)

print(f"✓ Processing complete!")
print(f"Output directory: {output_dir}")
print(f"- {len(frames)} input images")
print(f"- {len(prediction.depth)} depth visualizations")
