#!/bin/bash
# Process all sample surgery videos from ../SourceDatasets/sample_surgery_videos
# For each dataset: run DA3 inference (generate_3d_map.py) then point cloud visualization
set -e

BASE="../SourceDatasets/sample_surgery_videos/sample_surgery_videos"
OUT_ROOT="workspace/surgery_samples"
MAX_FRAMES=20
FPS=2.0
PROCESS_RES=504

mkdir -p "$OUT_ROOT"

echo "============================================================"
echo "Processing all sample surgery videos"
echo "Output root: $OUT_ROOT"
echo "============================================================"

# --- 1. Cholec80 (661M video, smallest — run first) ---
DATASET="Cholec80"
echo -e "\n>>> [$DATASET] Running inference..."
python3 generate_3d_map.py \
    --video "$BASE/Cholec80/video01.mp4" \
    --output-dir "$OUT_ROOT/$DATASET" \
    --max-frames $MAX_FRAMES --fps $FPS --process-res $PROCESS_RES

echo ">>> [$DATASET] Visualizing point clouds..."
python3 visualize_pointcloud.py \
    --npz "$OUT_ROOT/$DATASET/exports/mini_npz/results.npz" \
    --images "$OUT_ROOT/$DATASET/input_images" \
    --output "$OUT_ROOT/$DATASET/pc_vis" \
    

# --- 2. RARP-50 (1.2G video) ---
DATASET="RARP-50"
echo -e "\n>>> [$DATASET] Running inference..."
python3 generate_3d_map.py \
    --video "$BASE/RARP-50/video_left.avi" \
    --output-dir "$OUT_ROOT/$DATASET" \
    --max-frames $MAX_FRAMES --fps $FPS --process-res $PROCESS_RES

echo ">>> [$DATASET] Visualizing point clouds..."
python3 visualize_pointcloud.py \
    --npz "$OUT_ROOT/$DATASET/exports/mini_npz/results.npz" \
    --images "$OUT_ROOT/$DATASET/input_images" \
    --output "$OUT_ROOT/$DATASET/pc_vis" \
    

# --- 3. Endovis2018SubChallenge (149 frames, image directory) ---
DATASET="Endovis2018SubChallenge"
echo -e "\n>>> [$DATASET] Running inference..."
python3 generate_3d_map.py \
    --image-dir "$BASE/Endovis2018SubChallenge/seq_1/left_frames" \
    --output-dir "$OUT_ROOT/$DATASET" \
    --max-frames $MAX_FRAMES --step 1 --process-res $PROCESS_RES

echo ">>> [$DATASET] Visualizing point clouds..."
python3 visualize_pointcloud.py \
    --npz "$OUT_ROOT/$DATASET/exports/mini_npz/results.npz" \
    --images "$BASE/Endovis2018SubChallenge/seq_1/left_frames" \
    --output "$OUT_ROOT/$DATASET/pc_vis" \
    

# --- 4. TMVP-SurVideo (9807 jpg frames, image directory) ---
DATASET="TMVP-SurVideo"
echo -e "\n>>> [$DATASET] Running inference..."
python3 generate_3d_map.py \
    --image-dir "$BASE/TMVP-SurVideo/001" \
    --output-dir "$OUT_ROOT/$DATASET" \
    --max-frames $MAX_FRAMES --step 50 --process-res $PROCESS_RES

echo ">>> [$DATASET] Visualizing point clouds..."
python3 visualize_pointcloud.py \
    --npz "$OUT_ROOT/$DATASET/exports/mini_npz/results.npz" \
    --images "$BASE/TMVP-SurVideo/001" \
    --output "$OUT_ROOT/$DATASET/pc_vis" \
    

# --- 5. HeiChole (4.3G video) ---
DATASET="HeiChole"
echo -e "\n>>> [$DATASET] Running inference..."
python3 generate_3d_map.py \
    --video "$BASE/HeiChole/HeiChole1.mp4" \
    --output-dir "$OUT_ROOT/$DATASET" \
    --max-frames $MAX_FRAMES --fps $FPS --process-res $PROCESS_RES

echo ">>> [$DATASET] Visualizing point clouds..."
python3 visualize_pointcloud.py \
    --npz "$OUT_ROOT/$DATASET/exports/mini_npz/results.npz" \
    --images "$OUT_ROOT/$DATASET/input_images" \
    --output "$OUT_ROOT/$DATASET/pc_vis" \
    

# --- 6. AutoLabaro (7.5G video, largest — run last) ---
DATASET="AutoLabaro"
echo -e "\n>>> [$DATASET] Running inference..."
python3 generate_3d_map.py \
    --video "$BASE/AutoLabaro/01.mp4" \
    --output-dir "$OUT_ROOT/$DATASET" \
    --max-frames $MAX_FRAMES --fps $FPS --process-res $PROCESS_RES

echo ">>> [$DATASET] Visualizing point clouds..."
python3 visualize_pointcloud.py \
    --npz "$OUT_ROOT/$DATASET/exports/mini_npz/results.npz" \
    --images "$OUT_ROOT/$DATASET/input_images" \
    --output "$OUT_ROOT/$DATASET/pc_vis" \
    

echo -e "\n============================================================"
echo "All datasets processed! Results in: $OUT_ROOT/"
echo "============================================================"
