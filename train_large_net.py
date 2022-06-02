import argparse
import time

import numpy as np
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from once_for_all.elastic_nn.training.progressive_shrinking import validate
from once_for_all.networks.kws_net import KWSNetLarge
from once_for_all.run_manager.run_config import KWSRunConfig
from once_for_all.run_manager.run_manager import RunManager
from utils.common_tools import AverageMeter, list_mean


def train_one_large_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network

    # switch to train mode
    dynamic_net.train()
    nBatch = len(run_manager.run_config.train_loader)

    #nBatch = 256
    print("nBatch: ", nBatch)

    data_time = AverageMeter()
    losses = AverageMeter()


    best_acc = 0
    running_loss = 0.0
    total = 0
    correct = 0

    metric_dict = run_manager.get_metric_dict()
    with tqdm(
            total=nBatch,
            desc="Train Epoch #{}".format(epoch + 1),
            disable=False,
    ) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):  # for each minibatch
            data_time.update(time.time() - end)

            # Set learning rate
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
            # print("Image shape: ", images.shape)
            # print("Labels shape: ", labels.shape)

            target = labels

            # clean gradients
            dynamic_net.zero_grad()
            outputs = run_manager.net(images)
            loss = nn.CrossEntropyLoss()
            # loss = run_manager.train_criterion(outputs, labels)
            loss_type = "ce"

            # Compute training statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print information every 20 minibatches
            if i % 20 == 0:
                print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (
                    i + 1, nBatch, running_loss / 10, 100 * correct / total))
                running_loss = 0.0


            loss.backward()
            run_manager.optimizer.step()

            t.set_postfix(
                {
                    "loss": loss.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "seed": str(0),
                    # "seed": str(subnet_seed),
                    "str": "test",
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
    return loss, run_manager.get_metric_vals(metric_dict)



def train_large_net(run_manager, args, validate_func=None):
    for epoch in range(
            run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs
    ):
        train_loss, (train_top1, train_top5) = train_one_large_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
        )
        # print("train_loss", train_loss)


        val_loss, metr = run_manager.validate()
        print("val loss and metrics")
        print(val_loss)
        print(metr)

        # tmp_acc, _ =

        # Save best performing network
        # if tmp_acc > best_acc:
        #    best_acc = tmp_acc
        #    PATH = './model_acc_' + str(best_acc) + '.pth'
        #    PATH = os.path.join(save_path, PATH)
        #    torch.save(model.state_dict(), PATH)

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
# args.dynamic_batch_size = 256
args.n_epochs = 120
args.base_lr = 3e-2
args.warmup_epochs = 5
args.warmup_lr = -1
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.kd_ratio = 0

args.ft_extr_size = [(49, 10), (60, 2)]

args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 1

args.train_batch_size = 256

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
    run_manager.train_large_net(args)
