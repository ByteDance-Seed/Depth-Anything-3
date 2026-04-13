#!/bin/bash
set -e
OUT_ROOT="workspace/surgery_samples"

for DATASET in Cholec80 RARP-50 Endovis2018SubChallenge TMVP-SurVideo HeiChole AutoLabaro; do
    echo -e "\n>>> [$DATASET] Visualizing point clouds (all frames, 2x4 layout)..."
    
    # Remove old vis
    rm -rf "$OUT_ROOT/$DATASET/pc_vis"
    
    python3 visualize_pointcloud.py \
        --npz "$OUT_ROOT/$DATASET/exports/mini_npz/results.npz" \
        --config "$OUT_ROOT/$DATASET/config.json" \
        --output "$OUT_ROOT/$DATASET/pc_vis"
done

echo -e "\n============================================================"
echo "All visualizations regenerated!"
echo "============================================================"
