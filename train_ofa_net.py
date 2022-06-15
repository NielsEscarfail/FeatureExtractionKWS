import argparse
import time

import numpy as np
import os
import random
import torch

from once_for_all.elastic_nn.networks.ofa_kws_net import OFAKWSNet
from once_for_all.networks.kws_net import KWSNetLarge
from once_for_all.run_manager.run_config import KWSRunConfig
from once_for_all.run_manager.run_manager import RunManager

from once_for_all.elastic_nn.training.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="normal",  # kernel
    choices=[
        "normal",
        "kernel",
        "depth",
        "expand",
        "classifier_dim",
    ],
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--resume", action="store_true")
parser.add_argument("--ft_extr_type",
                    type=str,
                    default="mfcc",
                    choices=[
                        "mfcc",
                        "mel_spectrogram",
                        "linear_stft"
                    ]
)


args = parser.parse_args()

# Set up model parameters depending on the task TODO
if args.task == "normal":
    args.path = "exp/normal"
    args.dynamic_batch_size = 1
    args.n_epochs = 120  # 180 paper
    args.base_lr = 0.001  # 3e-2  # 1e-3  # 3e-2 - 2.6 paper -> .5-.7?
    args.warmup_epochs = 0  # 5
    args.warmup_lr = -1
    args.ks_list = "7"
    args.expand_list = "6"
    args.depth_list = "4"
elif args.task == "kernel":  # params ok
    args.path = "exp/normal2kernel"
    args.dynamic_batch_size = 1
    args.n_epochs = 120  # 120
    args.base_lr = 3e-2
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = "3,5,7"
    args.expand_list = "6"
    args.depth_list = "4"
elif args.task == "depth":  # all params below ok
    args.path = "exp/kernel2kernel_depth/phase%d" % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3  # 2.5e-3 - 0.08 paper
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "3,4"
    else:
        args.n_epochs = 120  # 125 (120 + 5)
        args.base_lr = 7.5e-3  # 7.5e-3 - 0.24 paper
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "2,3,4"
elif args.task == "expand":
    args.path = "exp/kernel_depth2kernel_depth_width/phase%d" % args.phase
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "4,6"
        args.depth_list = "2,3,4"
    else:
        args.n_epochs = 120  # 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "3,4,6"
        args.depth_list = "2,3,4"
else:
    raise NotImplementedError

# Other fixed parameters TODO

args.manual_seed = 0

args.lr_schedule_type = "cosine"

args.base_batch_size = 256
args.valid_size = .1

args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 5
args.print_frequency = 5

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = "tf"
# args.image_size = "128,160,192,224"
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
# args.base_stage_width = "proxyless"

args.width_mult_list = "1.0"
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

# args.kd_ratio = 1.0
args.kd_ratio = 0
args.kd_type = "ce"

# Set ft_extr_params_list depending on the ft_extr_type
if args.ft_extr_type == "mel_spectrogram":  # n_mels, win_len
    args.ft_extr_params_list = [(10, 40), (10, 60), (10, 80),
                                (20, 40), (20, 60), (20, 80),
                                (30, 40), (30, 60), (30, 80),
                                (40, 40), (40, 60), (40, 80)]

elif args.ft_extr_type == "mfcc":  # n_mfcc, win_len
    args.ft_extr_params_list = [(10, 40), (10, 60), (10, 80),
                                (20, 40), (20, 60), (20, 80),
                                (30, 40), (30, 60), (30, 80),
                                (40, 40), (40, 60), (40, 80)]

elif args.ft_extr_type == "linear_stft":
    args.ft_extr_params_list = [
        (1024, 40), (1024, 60), (1024, 80),
        (2048, 40), (2048, 60), (2048, 80)]


if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)
    start = time.time()

    # Initialize Horovod
    # hvd.init()

    num_gpus = torch.cuda.device_count()
    # num_gpus = 1
    print("Using %f gpus" % num_gpus)

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

    # args.teacher_path = "ofa_checkpoints/ofa_D4_E6_K7" # TODO
    args.teacher_path = "exp/normal/checkpoint/checkpoint.pth.tar"  # TODO

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4

    run_config = KWSRunConfig(
        **args.__dict__, num_replicas=num_gpus
    )

    # Print run config information
    print("Run config:")
    for k, v in run_config.config.items():
        print("\t%s: %s" % (k, v))

    # Build net from args
    args.width_mult_list = [
        float(width_mult) for width_mult in args.width_mult_list.split(",")
    ]
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )
    # Instantiate OFA KWS model
    net = OFAKWSNet(
        n_classes=12,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        # base_stage_width=args.base_stage_width,
        width_mult=args.width_mult_list,
        ks_list=args.ks_list,
        expand_ratio_list=args.expand_list,
        depth_list=args.depth_list,
    )

    # Instantiate largest KWS model possible
    if args.kd_ratio > 0:
        args.teacher_model = KWSNetLarge(
            n_classes=12,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=0,
            width_mult=1.0,
            ks=7,
            expand_ratio=6,
            depth_param=4,
        )
        if torch.cuda.is_available():
            args.teacher_model.cuda()

    """ RunManager """
    run_manager = RunManager(
        args.path,
        net,
        run_config
    )
    run_manager.save_config()

    # load teacher net weights
    if args.kd_ratio > 0:
        load_models(
            run_manager, args.teacher_model, model_path=args.teacher_path
        )

    """Training"""
    from once_for_all.elastic_nn.training.progressive_shrinking import (
        validate,
        train,
    )

    validate_func_dict = {
        "ft_extr_type": args.ft_extr_type,
        "ft_extr_params_list": args.ft_extr_params_list,  # "ft_extr_params_list": [(10, 40), (40, 40), (40, 80)],
        "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
        "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    print("Validation feature extraction type: ", validate_func_dict['ft_extr_type'])
    print("Validation feature extraction parameter search space: ", validate_func_dict['ft_extr_params_list'])

    if args.task == "normal":
        """args.ofa_checkpoint_path = "exp/normal/checkpoint/model_best.pth.tar"

        load_models(
            run_manager,
            run_manager.net,
            args.ofa_checkpoint_path,
        )
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
        print("Resuming training ")"""
        print("Start large net training")
        train(
            run_manager,
            args,
            lambda _run_manager, epoch, is_test: validate(
                _run_manager, epoch, is_test, **validate_func_dict
            ))

    elif args.task == "kernel":
        validate_func_dict["ks_list"] = sorted(args.ks_list)
        if run_manager.start_epoch == 0:

            args.ofa_checkpoint_path = "exp/normal/checkpoint/model_best.pth.tar"

            load_models(
                run_manager,
                run_manager.net,
                args.ofa_checkpoint_path,
            )
            run_manager.write_log(
                "%.3f\t%.3f\t%.3f\t%s"
                % validate(run_manager, is_test=True, **validate_func_dict),
                "valid",
            )
            print("Start elastic kernel training")
        else:
            assert args.resume
        train(
            run_manager,
            args,
            lambda _run_manager, epoch, is_test: validate(
                _run_manager, epoch, is_test, **validate_func_dict
            ),
        )

    elif args.task == "depth":
        from once_for_all.elastic_nn.training.progressive_shrinking import (
            train_elastic_depth,
        )

        if args.phase == 1:
            args.ofa_checkpoint_path = "exp/normal2kernel/checkpoint/model_best.pth.tar"
            # args.ofa_checkpoint_path = "/ofa_checkpoints/ofa_D4_E6_K357"
        else:
            args.ofa_checkpoint_path = "exp/kernel2kernel_depth/phase1/checkpoint/model_best.pth.tar"
            # args.ofa_checkpoint_path = "/ofa_checkpoints/ofa_D34_E6_K357"

        print("Start elastic depth training")

        train_elastic_depth(train, run_manager, args, validate_func_dict)

    elif args.task == "expand":
        from once_for_all.elastic_nn.training.progressive_shrinking import (
            train_elastic_expand,
        )

        if args.phase == 1:
            args.ofa_checkpoint_path = "exp/kernel2kernel_depth/phase2/checkpoint/model_best.pth.tar"
            # args.ofa_checkpoint_path = "/ofa_checkpoints/ofa_D234_E6_K357"
        else:
            args.ofa_checkpoint_path = "exp/kernel_depth2kernel_depth_width/phase1/checkpoint/model_best.pth.tar"
            # args.ofa_checkpoint_path = "ofa_checkpoints/ofa_D234_E46_K357"

        print("Start elastic expand training")

        train_elastic_expand(train, run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError

    print("Validating all resolutions:")
    run_manager.validate_all_resolution(is_test=True)

    end = time.time()
    print("Total run time: ", end - start)
