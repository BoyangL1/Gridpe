### ViT-S Imagenet 100
#### Pre-train
```bash
#!/bin/bash

# Common training settings
DATA_PATH="/home/admin/data/deng/imagenet100"
BATCH_SIZE=128
EPOCHS=300
LR=5e-4
WEIGHT_DECAY=0.03
INPUT_SIZE=224
OUTPUT_BASE="/home/admin/data/deng/eval_results"
EXTRA_ARGS="--smoothing 0.0 --reprob 0.0 --opt adamw --color-jitter 0.3 --drop 0.0 --drop-path 0.0 --unscale-lr --repeated-aug --ThreeAugment --eval-crop-ratio 1.0 --dist-eval"

# Model-specific settings
declare -A MODELS
MODELS["gridpe_deit_small_patch16_LS"]="gridpe_deit_small_patch16_LS"
MODELS["rope_mixed_deit_small_patch16_LS"]="rope_mixed_deit_small_patch16_LS"
MODELS["rope_axial_deit_small_patch16_LS"]="rope_axial_deit_small_patch16_LS"
MODELS["deit_small_patch16_LS"]="deit_small_patch16_LS"

for MODEL_NAME in "${!MODELS[@]}"; do
  OUTPUT_DIR="${OUTPUT_BASE}/${MODELS[$MODEL_NAME]}/pretrain"
  echo "========== Training: ${MODEL_NAME} =========="
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
    ${EXTRA_ARGS}
done
```

#### Evaluation
```bash
#!/bin/bash

data_path="/home/admin/data/deng/imagenet100"

# Input sizes to evaluate
input_sizes=(160 192 224 256 320 384 448 512)

# Models and corresponding checkpoint paths
declare -A models
models["gridpe_deit_small_patch16_LS"]="/home/admin/data/deng/eval_results/gridpe_deit_small_patch16_LS/pretrain/best_checkpoint.pth"
models["rope_mixed_deit_small_patch16_LS"]="/home/admin/data/deng/eval_results/rope_mixed_deit_small_patch16_LS/pretrain/best_checkpoint.pth"
models["rope_axial_deit_small_patch16_LS"]="/home/admin/data/deng/eval_results/rope_axial_deit_small_patch16_LS/pretrain/best_checkpoint.pth"
models["deit_small_patch16_LS"]="/home/admin/data/deng/eval_results/deit_small_patch16_LS/pretrain/best_checkpoint.pth"

for model_name in "${!models[@]}"
do
  checkpoint_file="${models[$model_name]}"
  for input_size in "${input_sizes[@]}"
  do
    echo "Distributed evaluating ${model_name} at input size ${input_size}"

    OMP_NUM_THREADS=1 python -m torch.distributed.launch \
      --nproc_per_node=8 \
      --nnodes=1 \
      --use_env main.py \
      --model ${model_name} \
      --finetune ${checkpoint_file} \
      --data-path ${data_path} \
      --output_dir /home/admin/data/deng/eval_results/${model_name}/evaluation/${input_size} \
      --batch-size ${batch_size} \
      --input-size ${input_size} \
      --eval \
      --eval-crop-ratio 1.0 \
      --dist-eval
  done
done
```


### ViT-L Imagenet 100
#### Pre-train
```bash
#!/bin/bash

# Common training settings
DATA_PATH="/home/admin/data/deng/imagenet100"
BATCH_SIZE=128
EPOCHS=300
LR=5e-4
WEIGHT_DECAY=0.03
INPUT_SIZE=224
OUTPUT_BASE="/home/admin/data/deng/eval_results_large"
EXTRA_ARGS="--smoothing 0.0 --reprob 0.0 --opt adamw --color-jitter 0.3 --drop 0.0 --drop-path 0.0 --unscale-lr --repeated-aug --ThreeAugment --eval-crop-ratio 1.0 --dist-eval"

# Model-specific settings
declare -A MODELS
MODELS["gridpe_deit_large_patch16_LS"]="gridpe_deit_large_patch16_LS_100"
MODELS["rope_mixed_deit_large_patch16_LS"]="rope_mixed_deit_large_patch16_LS"
MODELS["rope_axial_deit_large_patch16_LS"]="rope_axial_deit_large_patch16_LS"
MODELS["deit_large_patch16_LS"]="deit_large_patch16_LS"

for MODEL_NAME in "${!MODELS[@]}"; do
  OUTPUT_DIR="${OUTPUT_BASE}/${MODELS[$MODEL_NAME]}/pretrain"
  echo "========== Training: ${MODEL_NAME} =========="
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
    ${EXTRA_ARGS}
done
```

#### Evaluation
```bash
#!/bin/bash

data_path="/home/admin/data/deng/imagenet100"

# Input sizes to evaluate
input_sizes=(160 192 224 256 320 384 448 512)

# Models and corresponding checkpoint paths
declare -A models
models["gridpe_deit_large_patch16_LS"]="/home/admin/data/deng/eval_results_large/gridpe_deit_large_patch16_LS/pretrain/best_checkpoint.pth"
models["rope_mixed_deit_large_patch16_LS"]="/home/admin/data/deng/eval_results_large/rope_mixed_deit_large_patch16_LS/pretrain/best_checkpoint.pth"
models["rope_axial_deit_large_patch16_LS"]="/home/admin/data/deng/eval_results_large/rope_axial_deit_large_patch16_LS/pretrain/best_checkpoint.pth"
models["deit_large_patch16_LS"]="/home/admin/data/deng/eval_results_large/deit_large_patch16_LS/pretrain/best_checkpoint.pth"

for model_name in "${!models[@]}"
do
  checkpoint_file="${models[$model_name]}"
  for input_size in "${input_sizes[@]}"
  do
    echo "Distributed evaluating ${model_name} at input size ${input_size}"

    OMP_NUM_THREADS=1 python -m torch.distributed.launch \
      --nproc_per_node=8 \
      --nnodes=1 \
      --use_env main.py \
      --model ${model_name} \
      --finetune ${checkpoint_file} \
      --data-path ${data_path} \
      --output_dir /home/admin/data/deng/eval_results/${model_name}/evaluation/${input_size} \
      --batch-size ${batch_size} \
      --input-size ${input_size} \
      --eval \
      --eval-crop-ratio 1.0 \
      --dist-eval
  done
done
```