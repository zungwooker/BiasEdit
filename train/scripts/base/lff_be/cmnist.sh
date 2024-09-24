# Do for seed 0
for pct in 0.5pct 1pct 2pct 5pct
do
python train.py \
    --dataset cmnist \
    --pct $pct \
    --base \
    --train_lff_be \
    --projcode "Baselines CMNIST" \
    --run_name "lff_be" \
    --seed 0 \
    --wandb \
    --data_dir "/mnt/sdd/Debiasing/benchmarks" \
    --gpu_num 6
done

## Do for rest of seeds
# for pct in 0pct 0.5pct 1pct 2pct 5pct
# do
#     for seed in 1 2 3 4
#     do
#     python train.py \
#         --dataset cmnist \
#         --pct $pct \
#         --base \
#         --train_lff_be \
#         --projcode "Baselines CMNIST" \
#         --run_name "lff_be" \
#         --seed $seed \
#         --wandb \
#         --gpu_num ????
#     done
# done