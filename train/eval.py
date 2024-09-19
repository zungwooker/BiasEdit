import numpy as np
import torch
import random
from learner_ours import LearnerOurs
from learner_base import LearnerBase
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AAAI-2023-BiasEnsemble')

    # training
    parser.add_argument("--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument("--lr",help='learning rate',default=1e-3, type=float)
    parser.add_argument("--num_workers", help="workers number", default=4, type=int)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
    parser.add_argument('--gpu_num', required=True, type=str, help="GPU index")
    parser.add_argument("--num_steps", help="# of iterations", default= 500 * 100, type=int)
    parser.add_argument("--target_attr_idx", help="target_attr_idx", default= 0, type=int)
    parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default= 1, type=int)
    parser.add_argument("--dataset", help="data to train", default= 'cmnist', type=str) 
    parser.add_argument("--percent", help="percentage of conflict", default= "1pct", type=str)
    parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
    parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.7)
    parser.add_argument("--curr_step", help="curriculum steps", type=int, default= 0)
    parser.add_argument("--model", help="which network, [MLP, ResNet18]", default= 'MLP', type=str)

    # logging
    parser.add_argument('--wandb', action='store_true', help="wandb")
    parser.add_argument("--log_dir", help='path for saving model', default='./log', type=str)
    parser.add_argument("--data_dir", help='path for loading data', default='./dataset', type=str)
    parser.add_argument("--preproc_dir", type=str, default='none', help="Dir of preprocessed data.")
    parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)

    # experiment
    parser.add_argument("--ours", action="store_true", help="whether to train vanilla")
    parser.add_argument("--base", action="store_true", help="whether to train vanilla")
    parser.add_argument("--train_vanilla", action="store_true")
    parser.add_argument("--train_lff", action="store_true")
    parser.add_argument("--train_lff_be", action="store_true")

    parser.add_argument("--fix_randomseed", action="store_true", help="fix randomseed")
    parser.add_argument("--seed",  help="seed", type=int, default=0)
    parser.add_argument("--biased_model_train_iter", type=int, default=1000, help="biased_model_stop iteration")
    parser.add_argument("--biased_model_softmax_threshold", type=float, default=0.99, help="biased_model_softmax_threshold")
    parser.add_argument("--num_bias_models", type=int, default=5, help="number of bias models")
    parser.add_argument("--resnet_pretrained", action="store_true", help="use pretrained ResNet")
    parser.add_argument("--agreement", type=int, default=3, help="number of agreement")

    # eval
    parser.add_argument("--test_on_biased", action="store_true")

    args = parser.parse_args()
    
    # init learner
    if args.ours:
        learner = LearnerOurs(args)
    elif args.base:
        learner = LearnerBase(args)
    else:
        print("Choose one learner..")
        raise NotImplementedError

    learner.load_model_and_optimizer(model_path='/mnt/sdd/Debiasing/Adv_BiasEdit/BiasEdit/log/bffhq/Baselines bffhq 0pct/seed: 0 | Vanilla/best_model_d.th')
    learner.evaluate_on_bias()