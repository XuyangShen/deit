# torchrun \
#  --nproc_per_node=2 \
#  main.py \
#  --model norm_linear_vit_tiny \
#  --batch-size 2048 \
#  --num_workers 20 \
#  --data-path /nvme2/imagenet1k \
#   --output_dir output/norm_linear_vit_tiny \

# source activate /mnt/petrelfs/shenxuyang/miniconda3/envs/torch2.1.0
export PYTHONPATH=/mnt/petrelfs/share/pymc/new:$PYTHONPATH

spring.submit arun -s \
    -n 4 \
    -p speech \
    --gpu \
    --cpus-per-task 16 \
    --quotatype=auto \
    --job-name=pretrain_abs \
    --preempt-yes \
    "python main.py \
    --model norm_linear_vit_tiny_abs \
    --batch-size 400 \
    --data-path /mnt/petrelfs/shenxuyang/imagenet \
    --output_dir output/norm_linear_vit_tiny_abs_bs400 \
    "

# spring.submit arun -s \
#     -n 4 \
#     -p speech \
#     --gpu \
#     --cpus-per-task 16 \
#     --quotatype=auto \
#     --job-name=pretrain \
#     --preempt-yes \
#     "python main.py \
#     --model norm_linear_vit_tiny \
#     --batch-size 400 \
#     --data-path /mnt/petrelfs/shenxuyang/imagenet \
#     --output_dir output/norm_linear_vit_tiny_bs400 \
#     --clip-grad 5.0 \
#     "