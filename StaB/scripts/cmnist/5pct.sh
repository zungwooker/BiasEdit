python preprocess.py \
    --gpu_num 3 \
    --dataset cmnist \
    --percent 5pct \
    --root "/mnt/sdd/BiasEdit" \
    --preproc "preproc/preproc_MB" \
    --pretrained "/mnt/sdd/BiasEdit/pretrained/" \
    --extract_tags \
    --compute_tag_stats \
    --generate_gate