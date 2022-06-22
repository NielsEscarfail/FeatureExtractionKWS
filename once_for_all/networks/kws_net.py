import copy
import torch.nn as nn
from utils import MyNetwork
from utils.layers import set_layer_from_config, ResidualBlock, MBConvLayer, IdentityLayer, LinearLayer, \
    ConvLayer
from utils.pytorch_modules import MyGlobalAvgPool2d, make_divisible


class KWSNet(MyNetwork):

    def __init__(self, input_stem, blocks, classifier):
        super(KWSNet, self).__init__()

        self.input_stem = nn.ModuleList(input_stem)
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)  # global_avg_pool  # MyGlobalAvgPool2d(keep_dim=True)
        self.classifier = classifier

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)  # global average pooling
        x = x.view(x.size(0), -1)  # torch.flatten(x, 1)
        x = self.classifier(x)
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
            blocks.append(ResidualBlock.build_from_config(block_config))

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

    # NOT UPDATED YET - might not be needed
    @staticmethod
    def build_net_via_cfg():
        raise NotImplementedError

    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()
        for key in state_dict:
            if key not in current_state_dict:
                assert ".mobile_inverted_conv." in key
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            current_state_dict[new_key] = state_dict[key]
        super(KWSNet, self).load_state_dict(current_state_dict)


class KWSNetLarge(KWSNet):

    def __init__(
            self,
            n_classes=12,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            ks=7,
            depth=8,
            width_mult=2.0,
    ):

        input_channel = int(make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE))

        width_list = [64, 64, 64, 64, 64, 64, 64, 64]

        for i, width in enumerate(width_list):
            width_list[i] = int(make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE))

        # build input stem
        input_stem = [
            ConvLayer(
                in_channels=1,
                out_channels=input_channel,
                kernel_size=(9, 5),
                stride=2,
                use_bn=True,
                act_func="relu")
        ]

        # Set stride, activation function, and SE dim reduction
        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"]
        se_stages = [False, False, False, False, False, False]
        n_block_list = [1] + [depth] * 2

        # blocks
        self.block_group_info = []
        blocks = []
        _block_index = 1
        feature_dim = input_channel

        for n_block, width, s, act_func, use_se in zip(
                n_block_list,
                width_list,
                stride_stages,
                act_stages,
                se_stages,
        ):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = 1  # stride = s if i == 0 else 1
                conv = MBConvLayer(in_channels=feature_dim,
                                   out_channels=output_channel,
                                   kernel_size=ks,
                                   expand_ratio=1,
                                   stride=stride,
                                   act_func=act_func,
                                   use_se=use_se)

                shortcut = IdentityLayer(feature_dim,
                                         feature_dim) if stride == 1 and feature_dim == output_channel else None

                blocks.append(ResidualBlock(conv, shortcut))
                feature_dim = output_channel

        classifier = LinearLayer(
            feature_dim, n_classes, dropout_rate=dropout_rate
        )

        print("input stem, : ", input_stem)
        print("blocks , : ", blocks)
        print("classifier , : ", classifier)

        super(KWSNet, self).__init__(
            input_stem=input_stem, blocks=blocks, classifier=classifier
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @staticmethod
    def build_net_via_cfg():
        raise NotImplementedError
