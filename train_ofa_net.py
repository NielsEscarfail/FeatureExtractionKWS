import argparse
import time

import numpy as np
import os
import random
import torch

from once_for_all.elastic_nn.networks.ofa_kws_net import OFAKWSNet
from once_for_all.networks.kws_net import KWSNetLarge, KWSNet
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
    ],
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("--resume", action="store_true")
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
                    ]
                    )

args = parser.parse_args()

args.path = "exp/" + args.ft_extr_type
args.kd_ratio = 1.0
args.width_mult_list = "1.0"

if args.task == "normal":
    args.path += "/normal"
    args.dynamic_batch_size = 1
    args.n_epochs = 140  # 50  # 80  # 140  # 120  # 180 paper
    args.base_lr = 1e-3  # 1e-3  # 0.001 # 3e-2  # 0.001  # 3e-2  # 1e-3  # 3e-2 - 2.6 paper -> .5-.7?
    args.warmup_epochs = 5  # 5
    args.warmup_lr = -1
    args.ks_list = "7"  # 7
    args.depth_list = "4"
    args.expand_list = "3"
    args.kd_ratio = 0
elif args.task == "kernel":
    args.path += "/normal2kernel"
    args.dynamic_batch_size = 1
    args.n_epochs = 100  # 120
    args.base_lr = 1e-3  # 1e-3
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = "3,5,7"
    args.depth_list = "4"  # "4" 3 2 1
    args.expand_list = "3"

elif args.task == "depth":
    args.path += "/kernel2kernel_depth/phase%d" % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25  # 25
        args.base_lr = 1e-3  # 1e-3  # 2.5e-3 - 0.08 paper
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.depth_list = "3,4"  # "3,4"
        args.expand_list = "3"

    elif args.phase == 2:
        args.n_epochs = 25  # 120  # 125 (120 + 5)
        args.base_lr = 1e-3  # 1e-3  # 7.5e-3 - 0.24 paper
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.depth_list = "2,3,4"  # "2,3,4"
        args.expand_list = "3"

    else:
        args.n_epochs = 100  # 120  # 125 (120 + 5)
        args.base_lr = 1e-3  # 1e-3  # 7.5e-3 - 0.24 paper
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.depth_list = "1,2,3,4"
        args.expand_list = "3"


elif args.task == "expand":
    args.path += "/kernel_depth2kernel_depth_expand/phase%d" % args.phase
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25  # 25
        args.base_lr = 1e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.depth_list = "1,2,4,6"  # "2,3,4"
        args.expand_list = "3,2"

    elif args.phase == 2:
        args.n_epochs = 100  # 55 # 120
        args.base_lr = 1e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.depth_list = "1,2,3,4"
        args.expand_list = "3,2,1"

else:
    raise NotImplementedError

args.manual_seed = 0

args.base_batch_size = 512
args.valid_size = .1

args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 3
args.print_frequency = 5

args.n_worker = 8

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1

args.kd_type = "ce"

"""Set ft_extr_params_list depending on the ft_extr_type"""

if args.ft_extr_type == "mfcc":  # n_mfcc/n_mels, win_len
    """MFCC params, shape (n_mels, win_len), n_mfcc is fixed to 10.
    used:
        - [(40, 40)]
        - [(40, 30), (40, 40), (40, 50),
            (80, 30), (80, 30), (80, 30)] works but 80 is meh
        - [(40, 30), (40, 40), (40, 50)]
        - [(10, 30), (10, 40), (10, 50),
            (20, 30), (20, 40), (20, 50),
            (30, 30), (30, 40), (30, 50),
            (40, 30), (40, 40), (40, 50)]
    """
    args.ft_extr_params_list = [(10, 30), (10, 40), (10, 50),
                                (20, 30), (20, 40), (20, 50),
                                (30, 30), (30, 40), (30, 50),
                                (40, 30), (40, 40), (40, 50)]

elif args.ft_extr_type == "mel_spectrogram":
    """MelSpectrogram params, shape (n_mels, win_len)
    used:
        - [(10, 20), (10, 25), (10, 30)]
    """
    args.ft_extr_params_list = [(10, 20), (10, 25), (10, 30)]

elif args.ft_extr_type == "linear_stft":  # n_mels unused
    """Linear STFT params, shape (_, win_len)
    in progress: first dimension is unused
        used:
            - [(1, 10), (1, 20), (1, 30), (1, 40), (1, 50), (1, 60)]
            - (1024, 40), (1024, 60), (1024, 80),
            (2048, 40), (2048, 60), (2048, 80)]
    """
    args.ft_extr_params_list = [(1, 10), (1, 20), (1, 30), (1, 40), (1, 50), (1, 60)]

elif args.ft_extr_type == "lpcc":
    args.ft_extr_params_list = [7, 9, 11, 13, 15]

elif args.ft_extr_type == "plp":  # Mega slow?
    args.ft_extr_params_list = [10, 15, 20, 25, 30, 35, 40]

elif args.ft_extr_type == "ngcc":  # n_ceps/order, nfilts TODO in progress, might be dropped
    args.ft_extr_params_list = [(10, 24), (10, 48), (10, 64),
                                (20, 24), (20, 48), (20, 64),
                                (30, 24), (30, 48), (30, 64),
                                (40, 24), (40, 48), (40, 64)]

elif args.ft_extr_type == "raw":
    args.ft_extr_params_list = [(125, 128)]

if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)
    start = time.time()

    num_gpus = torch.cuda.device_count()
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

    args.teacher_path = "exp/" + args.ft_extr_type
    args.teacher_path += "/normal/checkpoint/checkpoint.pth.tar"

    # build run config from args
    args.lr_schedule_param = None
    args.lr_schedule_type = "cosine"
    args.opt_type = "adam"  # cosine, sgd, adam

    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4

    run_config = KWSRunConfig(**args.__dict__, num_replicas=num_gpus)

    # Print run config information
    print("Run config:")
    for k, v in run_config.config.items():
        print("\t%s: %s" % (k, v))

    # Build net from args
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.width_mult_list = [float(w) for w in args.width_mult_list.split(",")]

    # Instantiate OFA KWS model
    net = OFAKWSNet(
        n_classes=12,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        width_mult_list=args.width_mult_list,
        ks_list=args.ks_list,
        depth_list=args.depth_list,
        expand_ratio_list=args.expand_list
    )

    """ RunManager """
    run_manager = RunManager(
        args.path,
        net,
        run_config
    )
    run_manager.save_config()

    # Instantiate largest KWS model possible
    if args.kd_ratio > 0:

        args.teacher_model = KWSNetLarge(
            n_classes=12,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=0,
            ks=max(args.ks_list),
            depth=max(args.depth_list),
            width_mult=max(args.width_mult_list),
            expand_ratio=max(args.expand_list)
        )

        if torch.cuda.is_available():
            args.teacher_model.cuda()

        #  load teacher net weights
        load_models(run_manager, args.teacher_model, model_path=args.teacher_path)

        # Validate teacher net
        teach_validate_func_dict = {
            "ft_extr_type": args.ft_extr_type,
            "ft_extr_params_list": args.ft_extr_params_list,
            "width_mult_list": [max(args.width_mult_list)],
            "ks_list": [max(args.ks_list)],
            "depth_list": [max(net.depth_list)],
            "expand_list": [max(args.expand_list)],
        }
        print("Teacher validation feature extraction type: ", teach_validate_func_dict['ft_extr_type'])
        print("Teacher validation feature extraction parameter search space: ",
              teach_validate_func_dict['ft_extr_params_list'])
        run_manager.validate_all_resolution(is_test=True, net=args.teacher_model)

    """Training"""
    from once_for_all.elastic_nn.training.progressive_shrinking import (
        validate,
        train,
    )

    validate_func_dict = {
        "ft_extr_type": args.ft_extr_type,
        "ft_extr_params_list": [(10, 30), (10, 50),
                                (40, 30), (40, 50)],
        "width_mult_list": sorted({min(args.width_mult_list), max(args.width_mult_list)}),
        "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
        "expand_list": sorted({min(args.expand_list), max(args.expand_list)}),
    }
    print("Validation feature extraction type: ", validate_func_dict['ft_extr_type'])
    print("Validation feature extraction parameter search space: ", validate_func_dict['ft_extr_params_list'])

    args.ofa_checkpoint_path = "exp/" + args.ft_extr_type
    if args.task == "normal":
        # Uncomment to resume training large net only
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
            args.ofa_checkpoint_path += "/normal/checkpoint/model_best.pth.tar"

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
            args.ofa_checkpoint_path += "/normal2kernel/checkpoint/model_best.pth.tar"
        elif args.phase == 2:
            args.ofa_checkpoint_path += "/kernel2kernel_depth/phase1/checkpoint/model_best.pth.tar"
        else:
            args.ofa_checkpoint_path += "/kernel2kernel_depth/phase2/checkpoint/model_best.pth.tar"

        print("Start elastic depth training")

        train_elastic_depth(train, run_manager, args, validate_func_dict)

    elif args.task == "expand":
        from once_for_all.elastic_nn.training.progressive_shrinking import (
            train_elastic_expand,
        )

        if args.phase == 1:
            args.ofa_checkpoint_path += "/kernel2kernel_depth/phase2/checkpoint/model_best.pth.tar"
        elif args.phase == 2:
            args.ofa_checkpoint_path += "/kernel_depth2kernel_depth_expand/phase1/checkpoint/model_best.pth.tar"
        else:
            args.ofa_checkpoint_path += "/kernel_depth2kernel_depth_expand/phase2/checkpoint/model_best.pth.tar"

        print("Start elastic width training")

        train_elastic_expand(train, run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError

    print("Testing all resolutions and networks:")
    validate_func_dict = {
        "ft_extr_type": args.ft_extr_type,
        "ft_extr_params_list": args.ft_extr_params_list,
        "width_mult_list": net.width_mult_list,
        "ks_list": net.ks_list,
        "depth_list": net.depth_list,
        "expand_list": net.expand_list,
    }
    print("Validation dict: ", validate_func_dict)
    validate(run_manager, is_test=True)
    end = time.time()
    print("Total run time: ", end - start)
