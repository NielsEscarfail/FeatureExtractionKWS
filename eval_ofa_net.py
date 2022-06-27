"""
Randomly samples N subnets from a given OFAKWSNet.
Evaluates them and saves their performance results to a csv file.
"""

import os
import time

import torch
import argparse

from once_for_all.elastic_nn.networks.ofa_kws_net import OFAKWSNet
from once_for_all.elastic_nn.training.progressive_shrinking import load_models
from once_for_all.run_manager import RunManager, KWSRunConfig

parser = argparse.ArgumentParser()

parser.add_argument("--ft_extr_type",
                    type=str,
                    default="mfcc",
                    choices=[
                        "mfcc",
                        "mel_spectrogram",
                        "linear_stft",
                        "lpcc",
                        "plp",
                        "ngcc",
                        "raw"
                    ])

args = parser.parse_args()

# Fixed path parameters


# Set width_mult_list, ks_list, expand_list and depth_list
args.width_mult_list = "1.0"
args.ks_list = "3,5,7"
args.expand_list = "1,2,4,6"
args.depth_list = "2,3,4"

# Set ft_extr_params

"""Set ft_extr_params_list depending on the ft_extr_type"""

if args.ft_extr_type == "mfcc":  # n_mfcc/n_mels, win_len
    """MFCC params, shape (n_mels, win_len), n_mfcc is fixed to 10.
    used:
        - [(40, 40)]
        - [(40, 30), (40, 40), (40, 50),
          (80, 30), (80, 30), (80, 30)]
        - [(40, 30), (40, 40), (40, 50)]
    """
    args.ft_extr_params_list = [(40, 40)]


args.path = "eval/"
args.ofa_checkpoint_path = "exp/" + args.ft_extr_type
args.ofa_checkpoint_path += "/kernel_depth2kernel_depth_expand/phase2/checkpoint/model_best.pth.tar"


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

    ofa_net = OFAKWSNet(
        n_classes=12,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        width_mult=args.width_mult_list,
        ks_list=args.ks_list,
        expand_ratio_list=args.expand_list,
        depth_list=args.depth_list,
    )

    """ RunManager """
    run_manager = RunManager(
        args.path,
        ofa_net,
        run_config
    )

    """  Randomly sample a sub-network, 
    To manually set the sub-network: 
        ofa_net.set_active_subnet(ks=7, e=6, d=4) 
    """
    ofa_net.sample_active_subnet()
    subnet = ofa_net.get_active_subnet(preserve_weight=True)

    run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)

load_models(
    run_manager,
    run_manager.net,
    args.ofa_checkpoint_path,
)

""" Test sampled subnet """
run_config.data_provider.assign_active_ft_extr_params(args.ft_extr_params)
run_manager.reset_running_statistics(net=subnet)

print("Test subnet:")
print(subnet.module_str)

loss, (top1, top5) = run_manager.validate(net=subnet)
print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))
