# # Do for seed 0
# for pct in 0.5pct
# do
# python train.py \
#     --dataset cmnist \
#     --pct $pct \
#     --ours \
#     --train_lff \
#     --projcode "BiasEdit cmnist" \
#     --run_name "lff" \
#     --seed 0 \
#     --data_dir "/mnt/sdd/Debiasing/benchmarks" \
#     --preproc_dir "/mnt/sdd/Debiasing/preproc/preproc_MB" \
#     --gpu_num 1
# done

# Do for rest of seeds
for pct in 0.5pct
do
    for seed in 0 1 2
    do
    python train.py \
        --dataset cmnist \
        --pct $pct \
        --ours \
        --train_lff \
        --wandb \
        --projcode "BiasEdit cmnist" \
        --run_name "lff" \
        --seed $seed \
        --data_dir "/mnt/sdd/Debiasing/benchmarks" \
        --preproc_dir "/mnt/sdd/Debiasing/preproc/preproc_MB" \
        --gpu_num 1
    done
done