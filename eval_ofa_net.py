"""
Randomly samples N subnets from a given OFAKWSNet.
Evaluates them and saves their performance results to a csv file.
"""

import os
import random
import time

import numpy as np
import torch
import argparse

from once_for_all.elastic_nn.networks.ofa_kws_net import OFAKWSNet
from once_for_all.elastic_nn.training.progressive_shrinking import load_models
from once_for_all.evaluation.perf_dataset import PerformanceDataset
from once_for_all.run_manager import RunManager, KWSRunConfig

parser = argparse.ArgumentParser()

parser.add_argument("--ft_extr_type",
                    type=str,
                    default="mfcc",
                    choices=[
                        "mfcc",
                        "mel_spectrogram",
                    ])

parser.add_argument("--n_arch", type=int, default=5)
parser.add_argument('--use_csv', action='store_true')
parser.add_argument('--use_json', dest='use_csv', action='store_false')
parser.set_defaults(use_csv=True)

args = parser.parse_args()

"""Set width_mult_list, ks_list, expand_list and depth_list"""

args.width_mult_list = "1.0"
args.ks_list = "3,5,7"
args.expand_list = "1,2,3"
args.depth_list = "0,1,2,3,4"


"""Set ft_extr_params_list depending on the ft_extr_type"""

if args.ft_extr_type == "mfcc":  # n_mfcc/n_mels, win_len
    """MFCC params, shape (n_mels, win_len), n_mfcc is fixed to 10.
    We choose to fix n_mels to 10, 40, 80 in each runs,
    as OFA tends to learn only one n_mels configuration when mixing them.
    params used:
        - [(40, 40)]
        - [(10, 30), (10, 40), (10, 50)], n_bin_count=10
        - [(40, 30), (40, 40), (40, 50)], n_bin_count=10, 40
        - [(80, 30), (80, 40), (80, 50)], n_bin_count=10, 40, 80
        Experimental:
        - [(40, 40)]
        - [(40, 30), (40, 40), (40, 50),
            (80, 30), (80, 30), (80, 30)] works but 80 is meh
        - [(40, 30), (40, 40), (40, 50)]
        - [(10, 30), (10, 40), (10, 50),
            (20, 30), (20, 40), (20, 50),
            (30, 30), (30, 40), (30, 50),
            (40, 30), (40, 40), (40, 50)]
    """
    args.ft_extr_params_list = [(40, 30), (40, 40), (40, 50)]


# Fixed path parameters
args.path = "eval/"
args.ofa_checkpoint_path = "exp/" + args.ft_extr_type
args.ofa_checkpoint_path += "/kernel_depth2kernel_depth_expand/phase2/checkpoint/model_best.pth.tar"

# Other parameters
args.manual_seed = 0
args.n_worker = 8
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1

if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)
    start = time.time()

    num_gpus = torch.cuda.device_count()
    print("Using %f cuda devices" % num_gpus)

    # Set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # Cuda Setup
    if torch.cuda.is_available():
        # Pin GPU to be used to process local rank (one GPU per process)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.manual_seed)
        print('Using GPU.')
    else:
        print('Using CPU.')

    run_config = KWSRunConfig(**args.__dict__, num_replicas=num_gpus)
    # Print run config information
    print("Run config:")
    for k, v in run_config.config.items():
        print("\t%s: %s" % (k, v))

    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.width_mult_list = [float(w) for w in args.width_mult_list.split(",")]

    """Instantiate OFAKWSNet and load trained model"""
    ofa_net = OFAKWSNet(
        n_classes=12,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        width_mult_list=args.width_mult_list,
        ks_list=args.ks_list,
        expand_ratio_list=args.expand_list,
        depth_list=args.depth_list,
    )

    """ RunManager """
    run_manager = RunManager(
        args.path,
        ofa_net,
        run_config,
        init=False
    )

    load_models(
        run_manager,
        run_manager.net,
        args.ofa_checkpoint_path,
    )

    """
    Create & build the performance dataset
    build_dataset randomly samples n_arch subnets and saves their config, accuracy, n_params, flops, latency
    """
    performance_dataset = PerformanceDataset(args.path, use_csv=args.use_csv)
    # if not args.load:
    performance_dataset.build_dataset(run_manager, ofa_net,
                                      n_arch=args.n_arch, ft_extr_params_list=args.ft_extr_params_list)
