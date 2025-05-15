#!/bin/bash
clear

# Set common training parameters
NUM_POINTS=512
BATCH_SIZE=32
EPOCHS=100
LR=5e-5
NP=6  # number of processes for DDP

# Train PCT-GRID
echo "Training: PCT-GRID"
torchrun --nproc_per_node=$NP main_ddp.py --model pctgrid --exp_name train_pctgrid --num_points $NUM_POINTS --use_sgd False --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR

# # Train PCT-ROPE
echo "Training: PCT-ROPE"
torchrun --nproc_per_node=$NP main_ddp.py --model pctrope --exp_name train_pctrope --num_points $NUM_POINTS --use_sgd False --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR

# Train PCT
echo "Training: PCT"
torchrun --nproc_per_node=$NP main_ddp.py --model pct --exp_name train_pct --num_points $NUM_POINTS --use_sgd False --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR

# Train PCT-XYZ
echo "Training: PCT-XYZ"
torchrun --nproc_per_node=$NP main_ddp.py --model pctxyz --exp_name train_pctxyz --num_points $NUM_POINTS --use_sgd False --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR