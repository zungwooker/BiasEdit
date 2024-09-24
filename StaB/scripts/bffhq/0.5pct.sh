python preprocess.py \
    --gpu_num 6 \
    --dataset bffhq \
    --percent 0.5pct \
    --root "/mnt/sdd/BiasEdit" \
    --preproc "preproc/preproc_MB" \
    --pretrained "/mnt/sdd/BiasEdit/pretrained/" \
    --extract_tags \
    --compute_tag_stats \
    --generate_gate