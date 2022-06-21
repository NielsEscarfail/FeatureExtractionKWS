import os
import torch
import argparse

from once_for_all.elastic_nn.networks.ofa_kws_net import OFAKWSNet
from once_for_all.elastic_nn.training.progressive_shrinking import load_models
from once_for_all.run_manager import RunManager, KWSRunConfig

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.ft_extr_type = "mfcc"
args.ft_extr_params = [(40, 40)]

args.path = "eval/"
args.ofa_checkpoint_path = "exp/" + args.ft_extr_type
args.ofa_checkpoint_path += "/kernel_depth2kernel_depth_width/phase2/checkpoint/model_best.pth.tar"

args.n_epochs = 40  # 55 # 120
args.base_lr = 7.5e-3
args.warmup_epochs = 0
args.warmup_lr = -1
args.ks_list = "3,5,7"
args.expand_list = "1,2,4,6"
args.depth_list = "2,3,4"

run_config = KWSRunConfig(**args.__dict__, num_replicas=num_gpus)
ofa_net = OFAKWSNet(
    n_classes=12,
    bn_param=(args.bn_momentum, args.bn_eps),
    dropout_rate=args.dropout,
    width_mult=args.width_mult_list,
    ks_list=args.ks_list,
    expand_ratio_list=args.expand_list,
    depth_list=args.depth_list,
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
