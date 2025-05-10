#!/bin/bash

data_path="/home/admin/don/Gridpe/imagenet100"

# Input sizes to evaluate
input_sizes=(160 192 224 256 320 384 448 512)

# Models and corresponding base paths (checkpoint files will be dynamically set)
declare -A MODELS
MODELS["rope_mixed_deit_small_patch16_LS"]="rope_mixed_deit_small_patch16_LS"
MODELS["rope_axial_deit_small_patch16_LS"]="rope_axial_deit_small_patch16_LS"
MODELS["deit_small_patch16_LS"]="deit_small_patch16_LS"
MODELS["gridpe_deit_small_patch16_LS"]="gridpe_deit_small_patch16_LS"

# Loop through each model and num_head
for model_name in "${!MODELS[@]}"
do
  for num_head in 4
  do
    # Dynamically set the checkpoint path for each num_head
    checkpoint_file="/home/admin/don/Gridpe/eval_results/${model_name}_${num_head}/pretrain/best_checkpoint.pth"
    
    # Loop through input sizes
    for input_size in "${input_sizes[@]}"
    do
      echo "========== Evaluating ${model_name} with num_heads=${num_head} at input size ${input_size} =========="
      
      # Adaptive batch size based on input size
      if [ "$input_size" -lt 640 ]; then
        batch_size=16
      elif [ "$input_size" -lt 768 ]; then
        batch_size=64
      elif [ "$input_size" -lt 896 ]; then
        batch_size=32
      elif [ "$input_size" -lt 1024 ]; then
        batch_size=16
      else
        batch_size=8
      fi

      # Define the output directory
      output_dir="/home/admin/don/Gridpe/eval_results/${model_name}_${num_head}/evaluation/${input_size}"

      # Run distributed evaluation
      OMP_NUM_THREADS=1 python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=1 \
        --use_env main.py \
        --model ${model_name} \
        --finetune ${checkpoint_file} \
        --data-path ${data_path} \
        --output_dir ${output_dir} \
        --batch-size ${batch_size} \
        --input-size ${input_size} \
        --num-heads ${num_head} \
        --eval \
        --eval-crop-ratio 1.0 \
        --dist-eval
    done
  done
done