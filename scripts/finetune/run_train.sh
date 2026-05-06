#!/usr/bin/env bash
# Edit the values below, then: bash scripts/finetune/run_train.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- data ----
DATA_ROOT=/home/derek_austin/Depth-Anything-3/datasets/drone_v60_depth
TRAIN_SPLIT=train
VAL_SPLIT=valid                   # set to None if you don't have a val/ folder yet
SUBSET_SIZE=None                # None = use full training set
ORIG_RES=256
PROC_RES=504
CANONICAL_FOCAL=300.0
DATASET_FOCAL_AT_ORIG=300.0

# ---- model ----
PRETRAINED_ID=depth-anything/DA3METRIC-LARGE
FREEZE_SKY_HEAD=true            # true | false

# ---- loss ----
LOSS=log_l1                     # log_l1 | silog
SILOG_LAMBDA=0.15

# ---- optim ----
BATCH_SIZE=12
NUM_WORKERS=12
EPOCHS=20
LR=1e-4
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
AMP_DTYPE=bfloat16              # bfloat16 | float16 | float32

# ---- logging / checkpoint ----
LOG_EVERY=20
VAL_EVERY_STEPS=8000
VAL_MAX_BATCHES=20
CKPT_DIR=./checkpoints/drone_v60
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)_depth"
SEED=0

# ---- wandb ----
WANDB=true                      # true | false
WANDB_PROJECT=da3-drone-finetune
WANDB_ENTITY=None               # None to use default
WANDB_MODE=online               # online | offline | disabled
LOG_IMAGES_EVERY=0            # 0 to disable image panels

# ---- debug ----
DRY_RUN=false                   # true = stop after 2 steps

# ---- upload ----
UPLOAD_TO_GCS=true              # true | false  — zip + upload ckpt dir after training
GCS_BUCKET=gs://model-runs-cv-zeromark

# -------------------- build CLI args --------------------
ARGS=(
    --data-root "$DATA_ROOT"
    --train-split "$TRAIN_SPLIT"
    --val-split "$VAL_SPLIT"
    --orig-res "$ORIG_RES"
    --proc-res "$PROC_RES"
    --canonical-focal "$CANONICAL_FOCAL"
    --dataset-focal-at-orig "$DATASET_FOCAL_AT_ORIG"
    --pretrained-id "$PRETRAINED_ID"
    --loss "$LOSS"
    --silog-lambda "$SILOG_LAMBDA"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --grad-clip "$GRAD_CLIP"
    --amp-dtype "$AMP_DTYPE"
    --log-every "$LOG_EVERY"
    --val-every-steps "$VAL_EVERY_STEPS"
    --val-max-batches "$VAL_MAX_BATCHES"
    --ckpt-dir "$CKPT_DIR"
    --run-name "$RUN_NAME"
    --seed "$SEED"
    --wandb-project "$WANDB_PROJECT"
    --wandb-entity "$WANDB_ENTITY"
    --wandb-mode "$WANDB_MODE"
    --log-images-every "$LOG_IMAGES_EVERY"
)

[[ -n "${SUBSET_SIZE:-}" ]]         && ARGS+=(--subset-size "$SUBSET_SIZE")
[[ "$FREEZE_SKY_HEAD" == "false" ]] && ARGS+=(--no-freeze-sky-head)
[[ "$WANDB" == "false" ]]           && ARGS+=(--no-wandb)
[[ "$DRY_RUN" == "true" ]]          && ARGS+=(--dry-run)

.venv/bin/python -m scripts.finetune.train "${ARGS[@]}"

# -------------------- upload checkpoints --------------------
if [[ "$UPLOAD_TO_GCS" == "true" ]]; then
    RUN_DIR="$CKPT_DIR/$RUN_NAME"
    if [[ ! -d "$RUN_DIR" ]]; then
        echo "[upload] run dir not found: $RUN_DIR — skipping" >&2
        exit 0
    fi
    ARCHIVE="/tmp/${RUN_NAME}.zip"
    echo "[upload] zipping $RUN_DIR -> $ARCHIVE"
    (cd "$(dirname "$RUN_DIR")" && zip -r "$ARCHIVE" "$(basename "$RUN_DIR")")
    echo "[upload] gsutil cp $ARCHIVE $GCS_BUCKET/"
    gsutil cp "$ARCHIVE" "$GCS_BUCKET/"
    rm -f "$ARCHIVE"
    echo "[upload] done -> $GCS_BUCKET/${RUN_NAME}.zip"
fi
