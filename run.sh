torchrun \
 --nproc_per_node=2 \
 main.py \
 --model norm_linear_vit_tiny \
 --batch-size 2048 \
 --num_workers 20 \
 --data-path /nvme2/imagenet1k \
  --output_dir output/norm_linear_vit_tiny \