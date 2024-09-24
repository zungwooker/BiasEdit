import argparse
import os
import json
import time

from module import Bias2Tag, IP2P, TagStats
from utils import Timer, makedir_preprocessed, load_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", type=int, help="GPU number")
    parser.add_argument("--dataset", type=str, help="Dataset")
    parser.add_argument("--percent", type=str, help="Conflict ratio")
    parser.add_argument("--n_bias", required=False, type=int, help="num of bias candidates: k", default=1)
    parser.add_argument("--root", type=str, help="Parent path of benchmarks & preprocessed.")
    parser.add_argument("--preproc", type=str, help="Dir of preprocessed data.")
    parser.add_argument("--pretrained", type=str, help="Dir of pretrained weights.")
    parser.add_argument("--seed", type=int, help="Random seed for editing.", default=0)
    
    # Tag2Text
    parser.add_argument("--extract_tags", action="store_true")
    parser.add_argument("--tag2text_thres", type=float, help="tag2text thres", default=0.68)
    
    # Bart text classification
    parser.add_argument("--compute_tag_stats", action="store_true")
    parser.add_argument("--sim_thres", type=float, help="Label filtering tag similarity thres", default=0.95)
    
    # InstructPix2Pix
    parser.add_argument("--generate_gate", action="store_true")
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="module/ip2p/configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="module/ip2p/checkpoints/MagicBrush-epoch-52-step-4999.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--num_split", type=int, default=1)
    parser.add_argument("--split_idx", type=int, default=0)

    args = parser.parse_args()
    
    # Load class name
    class_name_path = os.path.join(args.root, 'benchmarks', args.dataset, 'class_name.json') # benchmarks/{dataset}/class_name.json
    class_name = load_json(class_name_path)
    
    # Make preprocessed dirs.
    makedir_preprocessed(args=args)
    
    # Make tag2text.json
    if args.extract_tags:
        bias2tag = Bias2Tag(args=args, class_name=class_name,)
        bias2tag.generate_tag_json()
    
    # Make tag_stats.json
    if args.compute_tag_stats:
        tag_stats = TagStats(args=args, class_name=class_name)
        tag_stats.generate_tag_stats()
        tag_stats.integrate_tag_stats()
        tag_stats.condition_bias()
        tag_stats.mix_bias()
        tag_stats.detect_biased_samples()
        # tag_stats.class_bias_stats() # For easy generation; need to be fixed

    # Need to be fixed
    # Generate images
    if args.generate_gate:
        ip2p = IP2P(args=args)
        ip2p.edit_images()
    
if __name__ == '__main__':
    main()