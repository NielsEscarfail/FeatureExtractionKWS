import os
import json
import time
import random
import torch
import torch.nn as nn


class DistributedRunManager:
    def __init__(
            self,
            path,
            net,
            run_config,
            hvd_compression,
            backward_steps=1,
            is_root=False,
            init=True,
    ):
        import horovod.torch as hvd

        self.path = path
        self.net = net
        self.run_config = run_config
        self.is_root = is_root

        self.best_acc = 0.0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        self.net.cuda()
        cudnn.benchmark = True
        if init and self.is_root:
            init_models(self.net, self.run_config.model_init)
        if self.is_root:
            # print net info
            net_info = get_net_info(self.net, self.run_config.data_provider.data_shape)
            with open("%s/net_info.txt" % self.path, "w") as fout:
                fout.write(json.dumps(net_info, indent=4) + "\n")
                try:
                    fout.write(self.net.module_str + "\n")
                except Exception:
                    fout.write("%s do not support `module_str`" % type(self.net))
                fout.write(
                    "%s\n" % self.run_config.data_provider.train.dataset.transform
                )
                fout.write(
                    "%s\n" % self.run_config.data_provider.test.dataset.transform
                )
                fout.write("%s\n" % self.net)

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
                self.net.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.net.get_parameters(
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
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.net.named_parameters(),
            compression=hvd_compression,
            backward_passes_per_step=backward_steps,
        )
