import copy

import torch
import torch.nn as nn

from utils import MyNetwork, make_divisible, MyGlobalAvgPool2d
from utils.layers import set_layer_from_config, ResidualBlock, PadConvResBlock, ConvLayer, \
    MBConvLayer, IdentityLayer, LinearLayer, LSTMLayer
from utils.pytorch_modules import MyGlobalAvgPool2d


class KWSNet(MyNetwork):

    def __init__(self, input_stem, blocks, classifier):
        super(KWSNet, self).__init__()

        self.input_stem = nn.ModuleList(input_stem)
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
        self.classifier = classifier

    def forward(self, x):
        # print("1 ", x.shape)
        for layer in self.input_stem:
            x = layer(x)
            # print("2 ", x.shape)
        for block in self.blocks:
            x = block(x)
            # print("3 ", x.shape)
        x = self.global_avg_pool(x)  # global average pooling
        # print("4 ", x.shape)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # print("5 ", x.shape)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            _str += layer.module_str + "\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": KWSNet.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        input_stem = []
        for layer_config in config["input_stem"]:
            input_stem.append(set_layer_from_config(layer_config))

        blocks = []
        for block_config in config["blocks"]:
            blocks.append(PadConvResBlock.build_from_config(block_config))  # TODO ADD OUR CUSTOM BLOCK

        classifier = set_layer_from_config(config["classifier"])

        net = KWSNet(input_stem, blocks, classifier)

        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)
        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if isinstance(m.conv, MBConvLayer) and isinstance(
                    m.shortcut, IdentityLayer
                ):
                    m.conv.point_linear.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks[1:], 1):
            if block.shortcut is None and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    @staticmethod  # TODO IN PROGRESS
    def build_net_via_cfg(input_stem_cfg, blocks_cfg, input_channel, n_classes, dropout_rate):
        """
        Possible future additions:
        Add a last_channel argument with a final_expand_layer + feature_mix_layer before classifier
        """
        # build input stem # TODO make that depend on input_stem_cfg
        input_stem = []
        for stage_id, input_stem_cfg_list in input_stem_cfg.items():
            for (
                k,
                mid_channel,
                out_channel,
                use_se,
                act_func,
                stride,
                expand_ratio,
            ) in input_stem_cfg_list:
                mb_conv = MBConvLayer(
                    input_channel,
                    out_channel,
                    k,
                    stride,
                    expand_ratio,
                    mid_channel,
                    act_func,
                    use_se,
                )
                input_stem.append(mb_conv)
                feature_dim = out_channel

        # build mobile blocks
        # feature_dim = out_channel
        blocks = []
        for stage_id, block_config_list in blocks_cfg.items():
            for (
                k,
                mid_channel,
                out_channel,
                use_se,
                act_func,
                stride,
                expand_ratio,
            ) in block_config_list:
                mb_conv = MBConvLayer(
                    feature_dim,
                    out_channel,
                    k,
                    stride,
                    expand_ratio,
                    mid_channel,
                    act_func,
                    use_se,
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel

        # classifier
        classifier = LinearLayer(feature_dim, n_classes, dropout_rate=dropout_rate)

        return input_stem, blocks, classifier

    @staticmethod  # TODO CHECK
    def adjust_cfg(
            cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None
    ):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != "0":
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != "0":
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != "0":
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [
                    copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)
                ]
                cfg[stage_id] = new_block_config_list
        return cfg

    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()
        for key in state_dict:
            current_state_dict[key] = state_dict[key]
        super(KWSNet, self).load_state_dict(current_state_dict)


class KWSNetLarge(KWSNet):
    def __init__(
            self,
            n_classes=1000,
            width_mult=1.0,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.2,
            ks=None,
            expand_ratio=None,
            depth_param=None,
            stage_width_list=None,
    ):
        input_channel = 1  # Todo see for largest net
        input_stem_cfg = {
            "0": [
                [1, 1, 16, False, "relu", 1, 1],
            ]
        }
        blocks_cfg = {  # Here largest depth = 6
            #   k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio
            #    k,     exp,    c,      se,         nl,         s,      e,
            "0": [
                [3, 16, 16, False, "relu", 1, 1],
            ],
            "1": [
                [3, 64, 24, False, "relu", 2, None],  # 4
                [3, 72, 24, False, "relu", 1, None],  # 3
            ],
            "2": [
                [5, 72, 40, True, "relu", 2, None],  # 3
                [5, 120, 40, True, "relu", 1, None],  # 3
                [5, 120, 40, True, "relu", 1, None],  # 3
            ],
            "3": [
                [3, 240, 80, False, "h_swish", 2, None],  # 6
                [3, 200, 80, False, "h_swish", 1, None],  # 2.5
                [3, 184, 80, False, "h_swish", 1, None],  # 2.3
                [3, 184, 80, False, "h_swish", 1, None],  # 2.3
            ],
            "4": [
                [3, 480, 112, True, "h_swish", 1, None],  # 6
                [3, 672, 112, True, "h_swish", 1, None],  # 6
            ],
            "5": [
                [5, 100, 80, True, "h_swish", 2, None],  # 6
                [5, 100, 80, True, "h_swish", 1, None],  # 6
                [5, 100, 80, True, "h_swish", 1, None],  # 6
            ],
        }

        blocks_cfg = self.adjust_cfg(blocks_cfg, ks, expand_ratio, depth_param, stage_width_list)
        """
        # width multiplier on mobile setting, change `exp: 1` and `c: 2`
        for stage_id, block_config_list in cfg.items():
            for block_config in block_config_list:
                if block_config[1] is not None:
                    block_config[1] = make_divisible(
                        block_config[1] * width_mult, MyNetwork.CHANNEL_DIVISIBLE
                    )
                block_config[2] = make_divisible(
                    block_config[2] * width_mult, MyNetwork.CHANNEL_DIVISIBLE
                )
        """
        (
            input_stem,
            blocks,
            classifier,
        ) = self.build_net_via_cfg(
            input_stem_cfg, blocks_cfg, input_channel, n_classes, dropout_rate)

        super(KWSNetLarge, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)
