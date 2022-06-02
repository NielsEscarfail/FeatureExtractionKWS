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
            base_stage_width=None,
            width_mult=1.0,
            ks_list=3,
            expand_ratio_list=6,
            depth_list=4,
    ):

        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()


        # Set stride, activation function, and SE dim reduction
        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"]
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = [16, 24, 40, 80, 112, 160, 960, 1280]

        input_channel, first_block_dim = width_list[0], width_list[1]

        # build input stem
        input_stem_width_list = [1]
        input_stem_stride_stages = [1]
        input_stem_act_stages = ["relu"]
        input_stem_se_stages = [False]
        input_stem_n_block_list = [1]

        feature_dim = first_block_dim

        input_stem = []
        for width, n_block, s, act_func, use_se in zip(
            input_stem_width_list,
            input_stem_n_block_list,
            input_stem_stride_stages,
            input_stem_act_stages,
            input_stem_se_stages
        ):
            output_channel = width

            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # input conv layer # TODO check if we want Residual or just MBConv
                input_block_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                )
                input_stem.append(input_block_conv)
                feature_dim = output_channel


        # First block
        first_block_conv = MBConvLayer(
            in_channels=input_channel,
            out_channels=first_block_dim,
            kernel_size=3,
            stride=stride_stages[0],
            expand_ratio=1,
            act_func=act_stages[0],
            use_se=se_stages[0],
        )
        first_block = ResidualBlock(
            first_block_conv,
            IdentityLayer(first_block_dim, first_block_dim)
            if input_channel == first_block_dim
            else None,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = first_block_dim

        for width, n_block, s, act_func, use_se in zip(
            width_list[2:],
            n_block_list[1:],
            stride_stages[1:],
            act_stages[1:],
            se_stages[1:],
        ):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
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

        # first block
        x = self.blocks[0](x)
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            _str += layer.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        _str += self.classifier.module_str + "\n"
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

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type="depth"):
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "expand_ratio":
            self.__dict__["_expand_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_expand_include_list"] = None
        self.__dict__["_ks_include_list"] = None

    def sample_active_subnet(self):
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )
        expand_candidates = (
            self.expand_ratio_list
            if self.__dict__.get("_expand_include_list", None) is None
            else self.__dict__["_expand_include_list"]
        )
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        input_stem = copy.deepcopy(self.input_stem)
        blocks = [copy.deepcopy(self.blocks[0])]

        classifier = copy.deepcopy(self.classifier)

        input_channel = blocks[0].conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    ResidualBlock(
                        self.blocks[idx].conv.get_active_subnet(
                            input_channel, preserve_weight
                        ),
                        copy.deepcopy(self.blocks[idx].shortcut),
                    )
                )
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = KWSNet(
            input_stem, blocks, classifier
        )
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        input_stem_config = self.input_stem.config
        first_block_config = self.blocks[0].config
        classifier_config = self.classifier.config

        block_config_list = [first_block_config]
        input_channel = first_block_config["conv"]["out_channels"]
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    {
                        "name": ResidualBlock.__name__,
                        "conv": self.blocks[idx].conv.get_active_subnet_config(
                            input_channel
                        ),
                        "shortcut": self.blocks[idx].shortcut.config
                        if self.blocks[idx].shortcut is not None
                        else None,
                    }
                )
                input_channel = self.blocks[idx].conv.active_out_channel
            block_config_list += stage_blocks

        return {
            "name": KWSNet.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": block_config_list,
            "classifier": classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)
