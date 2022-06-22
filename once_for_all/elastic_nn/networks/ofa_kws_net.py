import copy
import random

from once_for_all.elastic_nn.modules.dynamic_layers import (
    DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
)
from once_for_all.networks.kws_net import KWSNet
from utils import make_divisible, MyNetwork
from utils.common_tools import val2list
from utils.layers import (
    IdentityLayer,
    ResidualBlock,
)

__all__ = ["OFAKWSNet"]


class OFAKWSNet(KWSNet):
    def __init__(
            self,
            n_classes=12,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            ks_list=3,
            depth_list=4,
            width_mult_list=1.0,
            expand_ratio_list=1
    ):

        self.ks_list = val2list(ks_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.width_mult_list = val2list(width_mult_list)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)

        self.ks_list.sort()
        self.depth_list.sort()
        self.width_mult_list.sort()

        input_channel = [int(make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE))
                         for width_mult in self.width_mult_list]

        # build input stem
        input_stem = [
            DynamicConvLayer(
                in_channel_list=val2list(1),
                out_channel_list=input_channel,
                kernel_size=(9, 5),  # (9, 3) or (9, 5)
                stride=2,
                use_bn=True,
                act_func="relu")
        ]

        # Set stride, activation function, and SE dim reduction
        stride_stages = [1, 2, 2, 2]
        act_stages = ["relu", "relu", "relu", "relu"]
        se_stages = [False, False, False, False]
        n_block_list = [max(self.depth_list)] * 4  # block depth = 4 blocks
        stage_width_list = [64, 64, 64, 64]

        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [int(make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE))
                                   for width_mult in self.width_mult_list]

        # blocks
        blocks = []
        self.block_group_info = []
        _block_index = 0
        for n_block, width, s, act_func, use_se in zip(
                n_block_list,
                stage_width_list,
                stride_stages,
                act_stages,
                se_stages,
        ):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            for i in range(n_block):
                stride = s if i == 0 else 1  # stride = 1  #
                conv = DynamicMBConvLayer(in_channel_list=val2list(input_channel),
                                          out_channel_list=val2list(width),
                                          kernel_size_list=ks_list,
                                          expand_ratio_list=val2list(1),
                                          stride=stride,
                                          act_func=act_func,
                                          use_se=use_se)

                shortcut = IdentityLayer(input_channel,
                                         input_channel) if stride == 1 and input_channel == width else None

                blocks.append(ResidualBlock(conv, shortcut))
                input_channel = width

        classifier = DynamicLinearLayer(
            input_channel, n_classes, dropout_rate=dropout_rate
        )

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
            active_idx = block_idx[:depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            _str += layer.module_str + "\n"
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
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
            ks=max(self.ks_list), d=max(self.depth_list), w=max(self.width_mult_list)
        )

    def set_active_subnet(self, ks=None, d=None, w=None, **kwargs):

        ks = val2list(ks, len(self.blocks))
        depth = val2list(d, len(self.block_group_info))
        width_mult = val2list(w, len(self.blocks) + 1)

        # print("in set active subnet2: ks:%s, depth:%s, w:%s" % (ks, depth, width_mult))
        # print("in set active subnet2: ks:%s, depth:%s, width_mult:%s" % (ks, depth, width_mult)
        # print("self.blocks: ", self.blocks)
        # print("ks : ", ks)

        for block, k in zip(self.blocks, ks):  # this works
            if k is not None:
                block.conv.active_kernel_size = k

        for i, d in enumerate(depth):  # this works
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        if width_mult[0] is not None:
            self.input_stem[0].conv.active_out_channel = self.input_stem[0].active_out_channel = int(
                self.input_stem[0].out_channel_list[0] * width_mult[0])

        for i, w in enumerate(width_mult[1:]):
            if w is not None:
                self.blocks[i].active_out_channel = int(self.blocks[i].conv.out_channel_list[0] * w)

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
            ks_candidates = [ks_candidates for _ in range(len(self.blocks))]
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

        # sample width
        width_setting = []
        if not isinstance(width_candidates[0], list):
            width_candidates = [width_candidates for _ in range(len(self.blocks))]
        for w_set in width_candidates:
            w = random.choice(w_set)
            width_setting.append(w)

        arch_config = {"ks": ks_setting, "d": depth_setting, "w": width_setting}

        self.set_active_subnet(**arch_config)
        return arch_config

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

    def get_active_subnet(self, preserve_weight=True):
        input_stem = [copy.deepcopy(self.input_stem[0])]
        input_channel = self.input_stem[0].conv.active_out_channel  # .conv or not
        # blocks
        blocks = []
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

        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        _subnet = KWSNet(input_stem, blocks, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(1)]
        input_channel = self.input_stem_config["conv"]["out_channels"]

        block_config_list = []
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
                try:
                    input_channel = self.blocks[idx].conv.active_out_channel
                except Exception:
                    input_channel = self.blocks[idx].conv.out_channels
            block_config_list += stage_blocks

        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": KWSNet.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": block_config_list,
            "classifier": classifier_config,
        }

    @staticmethod
    def build_net_via_cfg():
        raise NotImplementedError

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.conv.re_organize_middle_weights(expand_ratio_stage)
