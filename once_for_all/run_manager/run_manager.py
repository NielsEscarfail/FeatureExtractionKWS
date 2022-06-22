# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import random
import time
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm

from utils import init_models
from utils.common_tools import AverageMeter, accuracy, write_log
from utils.pytorch_utils import get_net_info, cross_entropy_with_label_smoothing, cross_entropy_loss_with_soft_target


__all__ = ["RunManager"]


class RunManager:
    def __init__(
            self, path, net, run_config, init=True, measure_latency=None, no_gpu=False
    ):
        self.path = path
        self.net = net
        self.run_config = run_config

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device("cuda")
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        # initialize model (default)
        self.net = self.net.to(self.device)
        if init:
            init_models(self.net, run_config.model_init)

        # net info
        net_info = get_net_info(
            self.net, self.run_config.data_provider.data_shape, measure_latency, True
        )
        with open("%s/net_info.txt" % self.path, "w") as fout:
            fout.write(json.dumps(net_info, indent=4) + "\n")
            # noinspection PyBroadException
            try:
                fout.write(self.network.module_str + "\n")
            except Exception:
                pass
            fout.write("%s\n" % self.network)

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = (
                lambda pred, target: cross_entropy_with_label_smoothing(
                    pred, target, self.run_config.label_smoothing
                )
            )
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            net_params = [
                self.network.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.network.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            # noinspection PyBroadException
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.run_config.build_optimizer(net_params)

        self.net = torch.nn.DataParallel(self.net)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def network(self):
        return self.net.module if isinstance(self.net, nn.DataParallel) else self.net

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        write_log(self.logs_path, log_str, prefix, should_print, mode)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {"state_dict": self.network.state_dict()}

        if model_name is None:
            model_name = "checkpoint.pth.tar"

        checkpoint[
            "dataset"
        ] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, "latest.txt")
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, "w") as fout:
            fout.write(model_path + "\n")
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, "model_best.pth.tar")
            torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline()
                if model_fname[-1] == "\n":
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = "%s/checkpoint.pth.tar" % self.save_path  # changed, used to be "%s/checkpoint.pth.tar"
                with open(latest_fname, "w") as fout:
                    fout.write(model_fname + "\n")
            print("=> loading checkpoint '{}'".format(model_fname))
            checkpoint = torch.load(model_fname, map_location="cpu")
        except Exception:
            print("fail to load checkpoint from %s" % self.save_path)
            return {}

        self.network.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            self.best_acc = checkpoint["best_acc"]
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print("=> loaded checkpoint '{}'".format(model_fname))
        return checkpoint

    def save_config(self, extra_run_config=None, extra_net_config=None):
        """dump run_config and net_config to the model_folder"""
        run_save_path = os.path.join(self.path, "run.config")
        if not os.path.isfile(run_save_path):
            run_config = self.run_config.config
            if extra_run_config is not None:
                run_config.update(extra_run_config)
            json.dump(run_config, open(run_save_path, "w"), indent=4)
            print("Run configs dump to %s" % run_save_path)

        try:
            net_save_path = os.path.join(self.path, "net.config")
            net_config = self.network.config
            if extra_net_config is not None:
                net_config.update(extra_net_config)
            json.dump(net_config, open(net_save_path, "w"), indent=4)
            print("Network configs dump to %s" % net_save_path)
        except Exception:
            print("%s do not support net config" % type(self.network))

    """ metric related """

    def get_metric_dict(self):
        return {
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0].item(), output.size(0))
        metric_dict["top5"].update(acc5[0].item(), output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg for key in metric_dict}
        else:
            return [metric_dict[key].avg for key in metric_dict]

    def get_metric_names(self):
        return "top1", "top5"

    """ train and test """

    def validate(
            self,
            epoch=0,
            is_test=False,
            run_str="",
            net=None,
            data_loader=None,
            no_logs=False,
            train_mode=False,
    ):
        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = (
                self.run_config.test_loader if is_test else self.run_config.valid_loader
            )

        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            with tqdm(
                    total=len(data_loader),
                    desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                    disable=no_logs,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    self.update_metric(metric_dict, output, labels)

                    losses.update(loss.item(), images.size(0))
                    t.set_postfix(
                        {
                            "loss": losses.avg,
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "ft_extr_type": self.run_config.data_provider.active_ft_extr_type,
                            "ft_extr_params": self.run_config.data_provider.active_ft_extr_params,
                            "R": tuple(images.shape[2:]),
                        }
                    )
                    t.update(1)
        return losses.avg, self.get_metric_vals(metric_dict)

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):

        if net is None:
            net = self.network
        if isinstance(self.run_config.data_provider.ft_extr_params_list, list):
            ft_extr_params_list, loss_list, top1_list, top5_list = [], [], [], []
            for ft_extr_params in self.run_config.data_provider.ft_extr_params_list:
                ft_extr_params_list.append(ft_extr_params)
                self.run_config.data_provider.assign_active_ft_extr_params(ft_extr_params)
                self.reset_running_statistics(net=net)
                loss, (top1, top5) = self.validate(epoch, is_test, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return ft_extr_params_list, loss_list, top1_list, top5_list
        else:
            loss, (top1, top5) = self.validate(epoch, is_test, net=net)
            return (
                [self.run_config.data_provider.ft_extr_params_list],
                [loss],
                [top1],
                [top5],
            )

    def reset_running_statistics(
            self, net=None, subset_size=2000, subset_batch_size=200, data_loader=None
    ):
        from once_for_all.elastic_nn.utils import set_running_statistics

        if net is None:
            net = self.network
        if data_loader is None:
            data_loader = self.run_config.random_sub_train_loader(
                subset_size, subset_batch_size
            )
        set_running_statistics(net, data_loader)
