import argparse
import time

import numpy as np
import os
import random
import torch
from tqdm import tqdm

from once_for_all.networks.kws_net import KWSNetLarge
from once_for_all.run_manager.run_config import KWSRunConfig
from once_for_all.run_manager.run_manager import RunManager
from utils.common_tools import AverageMeter


def train_one_large_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network

    # switch to train mode
    dynamic_net.train()
    # nBatch = len(run_manager.run_config.train_loader)
    nBatch = 16

    data_time = AverageMeter()
    losses = AverageMeter()
    metric_dict = run_manager.get_metric_dict()

    with tqdm(
            total=nBatch,
            desc="Train Epoch #{}".format(epoch + 1),
            disable=False,
    ) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # clean gradients
            dynamic_net.zero_grad()

            loss_of_subnets = []
            # compute output
            subnet_str = ""
            for _ in range(args.dynamic_batch_size):

                output = run_manager.net(images)
                if args.kd_ratio == 0:
                    loss = run_manager.train_criterion(output, labels)
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(
                            output, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + run_manager.train_criterion(
                        output, labels
                    )
                    loss_type = "%.1fkd-%s & ce" % (args.kd_ratio, args.kd_type)

                # measure accuracy and record loss
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)

                loss.backward()
            run_manager.optimizer.step()

            losses.update(list_mean(loss_of_subnets), images.size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "seed": str(subnet_seed),
                    "str": subnet_str,
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train_large_net(run_manager, args, validate_func=None):
    for epoch in range(
            run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs
    ):
        train_loss, (train_top1, train_top5) = train_one_large_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
        )

        if (epoch + 1) % args.validation_frequency == 0:
            val_loss, val_acc, val_acc5, _val_log = validate_func(
                run_manager, epoch=epoch, is_test=False
            )
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            if not distributed or run_manager.is_root:
                val_log = (
                    "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                        epoch + 1 - args.warmup_epochs,
                        run_manager.run_config.n_epochs,
                        val_loss,
                        val_acc,
                        run_manager.best_acc,
                    )
                )
                val_log += ", Train top-1 {top1:.3f}, Train loss {loss:.3f}\t".format(
                    top1=train_top1, loss=train_loss
                )
                val_log += _val_log
                run_manager.write_log(val_log, "valid", should_print=False)

                run_manager.save_model(
                    {
                        "epoch": epoch,
                        "best_acc": run_manager.best_acc,
                        "optimizer": run_manager.optimizer.state_dict(),
                        "state_dict": run_manager.network.state_dict(),
                    },
                    is_best=is_best,
                )

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="depth",
    choices=[
        "kernel",
        "depth",
        "expand",
        "classifier_dim",
    ],
)

args = parser.parse_args()
args.path = "testlarge"
args.dynamic_batch_size = 1
args.n_epochs = 120
args.base_lr = 3e-2
args.warmup_epochs = 5
args.warmup_lr = -1
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1

if __name__ == "__main__":
    device = torch.device('cpu')

    run_config = KWSRunConfig(
        **args.__dict__
    )
    print("Run config:")
    for k, v in run_config.config.items():
        print("\t%s: %s" % (k, v))

    # Instantiate Large KWS model
    net = KWSNetLarge(
        n_classes=12,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=0,
        width_mult=1.0,
        ks=7,
        expand_ratio=6,
        depth_param=4,
    )

    """ RunManager """
    run_manager = RunManager(
        args.path,
        net,
        run_config
    )

    run_manager.save_config()

    """Training"""
    train_large_net(run_manager, args)
