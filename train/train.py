import numpy as np
import torch
import random
from learner_ours import LearnerOurs
from learner_base import LearnerBase
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVPR-2025-BiasEdit')

    # Hardware
    parser.add_argument("--num_workers", help="workers number", default=4, type=int)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
    parser.add_argument('--gpu_num', required=True, type=str, help="GPU index")
    
    # Data
    parser.add_argument("--dataset", required=True, help="data to train", type=str)
    parser.add_argument("--data_dir", required=True, help='path for loading data', type=str)
    parser.add_argument("--preproc_dir", type=str, default='none', help="Dir of preprocessed data.")
    parser.add_argument("--pct", required=True, help="percentage of conflict", type=str)
    
    parser.add_argument("--target_attr_idx", help="target_attr_idx", default= 0, type=int)
    parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default= 1, type=int)

    # logging
    parser.add_argument('--wandb', action='store_true', help="wandb")
    parser.add_argument("--projcode", help='project name of w & b', type=str)
    parser.add_argument("--run_name", help='run name of w & b', type=str)
    parser.add_argument("--log_dir", help='path for saving model', default='./log', type=str)
    parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)

    # Training Details
    parser.add_argument("--num_steps", help="# of iterations", default= 500 * 100, type=int)
    parser.add_argument("--ours", action="store_true", help="whether to train ours")
    parser.add_argument("--base", action="store_true", help="whether to train base")
    parser.add_argument("--train_vanilla", action="store_true")
    parser.add_argument("--train_lff", action="store_true")
    parser.add_argument("--train_lff_be", action="store_true")
    
    # Extra settings
    parser.add_argument("--mixup", action="store_true", help="whether to mixup generated BC with BA")
    parser.add_argument("--exchange", action="store_true", help="whether to exchange generated BC with BA")
    parser.add_argument("--b_for_entire", action="store_true", help="whether to set dataset for f_b")
    
    # Module details
    parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
    parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.7)
    parser.add_argument("--curr_step", help="curriculum steps", type=int, default= 0)
    
    # BiasEnsemble
    parser.add_argument("--biased_model_train_iter", type=int, default=1000, help="biased_model_stop iteration")
    parser.add_argument("--biased_model_softmax_threshold", type=float, default=0.99, help="biased_model_softmax_threshold")
    parser.add_argument("--num_bias_models", type=int, default=5, help="number of bias models")
    parser.add_argument("--agreement", type=int, default=3, help="number of agreement")

    # Reproducing
    parser.add_argument("--seed", required=True, help="seed", type=int)


    args = parser.parse_args()
    
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # init learner
    if args.ours:
        learner = LearnerOurs(args)
    elif args.base:
        learner = LearnerBase(args)
    else:
        print("Choose one learner..")
        raise NotImplementedError

    # actual training
    print('Training starts ...')

    learner.wandb_switch('start')
    if args.train_vanilla:
        learner.train_vanilla(args=args)
    elif args.train_lff:
        learner.train_lff(args=args)
    elif args.train_lff_be:
        learner.train_lff_be(args=args)
    else:
        print('choose one of the two options ...')
        import sys
        sys.exit(0)
    learner.wandb_switch('finish')