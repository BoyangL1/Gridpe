#!/bin/bash

# Common training settings
DATA_PATH="/home/admin/don/Gridpe/imagenet100"
BATCH_SIZE=32
EPOCHS=150
LR=5e-4
WEIGHT_DECAY=0.03
INPUT_SIZE=224
OUTPUT_BASE="/home/admin/don/Gridpe/eval_results"
EXTRA_ARGS="--smoothing 0.0 --reprob 0.0 --opt adamw --color-jitter 0.3 --drop 0.0 --drop-path 0.0 --unscale-lr --repeated-aug --ThreeAugment --eval-crop-ratio 1.0 --dist-eval"
NUM_HEADS=(4)

# Model-specific settings
declare -A MODELS
MODELS["deit_small_patch16_LS"]="deit_small_patch16_LS"
MODELS["rope_mixed_deit_small_patch16_LS"]="rope_mixed_deit_small_patch16_LS"
MODELS["rope_axial_deit_small_patch16_LS"]="rope_axial_deit_small_patch16_LS"
MODELS["gridpe_deit_small_patch16_LS"]="gridpe_deit_small_patch16_LS"

# Loop over the models and num_heads
for MODEL_NAME in "${!MODELS[@]}"; do
  for NUM_HEAD in "${NUM_HEADS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/${MODELS[$MODEL_NAME]}_${NUM_HEAD}/pretrain"
    echo "========== Training: ${MODEL_NAME} with num_heads=${NUM_HEAD} =========="
    echo "Output: ${OUTPUT_DIR}"
    
    OMP_NUM_THREADS=1 python -m torch.distributed.launch \
      --nproc_per_node=8 \
      --nnodes=1 \
      --use_env main.py \
      --model ${MODEL_NAME} \
      --data-path ${DATA_PATH} \
      --output_dir ${OUTPUT_DIR} \
      --batch-size ${BATCH_SIZE} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --weight-decay ${WEIGHT_DECAY} \
      --input-size ${INPUT_SIZE} \
      --num-heads ${NUM_HEAD} \
      ${EXTRA_ARGS}
  done
done