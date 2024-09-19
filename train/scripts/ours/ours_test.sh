python train.py \
    --dataset cmnist \
    --pct 0.5 \
    --ours \
    --train_lff \
    --projcode "OURSTEST CMNIST" \
    --run_name "lff" \
    --seed 0 \
    --wandb \
    --data_dir "/mnt/sdd/Debiasing/benchmarks" \
    --gpu_num 6

python train.py \
    --dataset cmnist \
    --pct 0.5 \
    --ours \
    --train_lff_be \
    --projcode "OURSTEST CMNIST" \
    --run_name "lff_be" \
    --seed 0 \
    --wandb \
    --data_dir "/mnt/sdd/Debiasing/benchmarks" \
    --gpu_num 6