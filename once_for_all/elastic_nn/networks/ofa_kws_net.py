import copy
import random

from once_for_all.elastic_nn.modules.dynamic_layers import (
    DynamicMBConvLayer, DynamicConvLayer
)
from utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
)
from once_for_all.networks.kws_net import KWSNet
from utils.common_tools import val2list

__all__ = ["OFAKWSNet"]


class OFAKWSNet(KWSNet):
    def __init__(
            self,
            n_classes=12,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            width_mult_list=1.0,
            ks_list=3,
            depth_list=4,
    ):

        self.ks_list = val2list(ks_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.width_mult_list = val2list(width_mult_list)

        self.ks_list.sort()
        self.depth_list.sort()
        self.width_mult_list.sort()

        # Set stride, activation function, and SE dim reduction
        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"]
        bn_stages = [True, True, True, True, True, True]
        n_block_list = [1] + [max(self.depth_list)] * 5

        width_list = [64, 64, 64, 64, 64, 64]
        for i, width in enumerate(width_list):
            width_list[i] = [width * width_mult for width_mult in self.width_mult_list]

        input_channel = width_list[0]

        # build input stem
        input_stem = [
            DynamicConvLayer(
                in_channel_list=["1"],
                out_channel_list=input_channel,
                kernel_size=3,
                stride=2,
                use_bn=True,
                act_func="relu", )
        ]

        # blocks
        self.block_group_info = []
        blocks = []
        _block_index = 1
        feature_dim = input_channel

        for n_block, width, s, act_func, use_bn in zip(
                n_block_list,
                width_list,
                stride_stages,
                act_stages,
                bn_stages,
        ):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = s if i == 0 else 1
                mobile_inverted_conv = DynamicConvLayer(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_bn=use_bn,
                )
                shortcut = IdentityLayer(feature_dim,
                                         feature_dim) if stride == 1 and feature_dim == output_channel else None

                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        classifier = LinearLayer(feature_dim, n_classes, dropout_rate=dropout_rate)

        super(OFAKWSNet, self).__init__(
            input_stem, blocks, classifier
        )

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    @staticmethod
    def name():
        return "OFAKWSNet"

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            _str += layer.module_str + "\n"
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"

        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": OFAKWSNet.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if ".mobile_inverted_conv." in key:
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif ".bn.bn." in new_key:
                new_key = new_key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in new_key:
                new_key = new_key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in new_key:
                new_key = new_key.replace(".linear.linear.", ".linear.")
            ##############################################################################
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAKWSNet, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(
            ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list)
        )

    def set_active_subnet(self, ks=None, d=None, w=None, **kwargs):
        ks = val2list(ks, len(self.blocks))
        depth = val2list(d, len(self.block_group_info) + 1)
        width_mult = val2list(w, len(self.width_mult_list) + 2)

        if width_mult[0] is not None:
            self.input_stem[1].conv.active_out_channel = self.input_stem[
                0
            ].active_out_channel = self.input_stem[0].out_channel_list[width_mult[0]]
        if width_mult[1] is not None:
            self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[
                width_mult[1]
            ]

        for stage_id, (block_idx, d, w) in enumerate(
            zip(self.grouped_block_index, depth[1:], width_mult[2:])
        ):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[
                        idx
                    ].out_channel_list[w]

        for block, k in zip(self.blocks[1:], ks):
            if k is not None:
                block.conv.active_kernel_size = k

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type="depth"):
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "width_mult":
            self.__dict__["_width_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_ks_include_list"] = None
        self.__dict__["_width_include_list"] = None

    def sample_active_subnet(self):
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )
        width_candidates = (
            self.width_mult_list
            if self.__dict__.get("_width_include_list", None) is None
            else self.__dict__["_width_include_list"]
        )

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        # sample width mult
        width_setting = []
        if not isinstance(width_candidates[0], list):
            width_candidates = [width_candidates for _ in range(len(self.blocks) - 1)]
        for w_set in width_candidates:
            w = random.choice(w_set)
            width_setting.append(w)

        self.set_active_subnet(ks_setting, depth_setting, width_setting)

        return {
            "ks": ks_setting,
            "d": depth_setting,
            "w": width_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        input_stem = copy.deepcopy(self.input_stem)
        blocks = [copy.deepcopy(self.blocks[0])]

        input_channel = self.input_stem[0].active_out_channel
        # blocks
        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(
                    self.blocks[idx].get_active_subnet(input_channel, preserve_weight)
                )
                input_channel = self.blocks[idx].active_out_channel

        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        _subnet = KWSNet(input_stem, blocks, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        input_channel = self.input_stem[2].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(
                    self.blocks[idx].get_active_subnet_config(input_channel)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": KWSNet.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": blocks_config,
            "classifier": classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)
