#TEST
clear
# PCT
for num in $(seq 256 128 2048)
do
    echo "Running evaluation with --num_points=${num}"
    torchrun --nproc_per_node=1 main_ddp.py \
        --exp_name test_pct_${num} \
        --num_points ${num} \
        --model pct \
        --use_sgd False \
        --eval True \
        --model_path checkpoints/train_pct/models/model.t7 \
        --test_batch_size 8
done

# PCT_xyz
for num in $(seq 256 128 2048)
do
    echo "Running evaluation with --num_points=${num}"
    torchrun --nproc_per_node=1 main_ddp.py \
        --exp_name test_pctxyz_${num} \
        --num_points ${num} \
        --model pctxyz \
        --use_sgd False \
        --eval True \
        --model_path checkpoints/train_pctxyz/models/model.t7 \
        --test_batch_size 8
done

# PCT_grid
for num in $(seq 256 128 2048)
do
    echo "Running evaluation with --num_points=${num}"
    torchrun --nproc_per_node=1 main_ddp.py \
        --exp_name test_pctgrid_${num} \
        --num_points ${num} \
        --model pctgrid \
        --use_sgd False \
        --eval True \
        --model_path checkpoints/train_pctgrid/models/model.t7 \
        --test_batch_size 8
done

# PCT_rope
for num in $(seq 256 128 2048)
do
    echo "Running evaluation with --num_points=${num}"
    torchrun --nproc_per_node=1 main_ddp.py \
        --exp_name test_pctrope_${num} \
        --num_points ${num} \
        --model pctrope \
        --use_sgd False \
        --eval True \
        --model_path checkpoints/train_pctrope/models/model.t7 \
        --test_batch_size 8
done