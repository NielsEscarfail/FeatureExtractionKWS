# Here implementation as in Nas Bench ASR
# LSTM + CELLS
# cf https://github.com/mit-han-lab/once-for-all/blob/4451593507b0f48a7854763adfe7785705abdd78/ofa/imagenet_classification/elastic_nn/networks/ofa_mbv3.py

# Contains MACRO ARCHITECTURE ON THE ASR NET I think, (fixed params)
# It is also therefore possible to have multiple input shapes (using make_divisible and adjusting widths)
from once_for_all.networks.nasasr_net import NASASRNet


class OFAKWSNet(NASASRNet):
    def __init__(self):
        pass



class OFAMobileNetV3(MobileNetV3):
    def __init__(
        self,
        n_classes=1000,
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


        class OFAResNets(ResNets):
            def __init__(
                    self,
                    n_classes=1000,
                    bn_param=(0.1, 1e-5),
                    dropout_rate=0,
                    depth_list=2,
                    expand_ratio_list=0.25,
                    width_mult_list=1.0,
            ):
                self.depth_list = val2list(depth_list)
                self.expand_ratio_list = val2list(expand_ratio_list)
                self.width_mult_list = val2list(width_mult_list)
                # sort
                self.depth_list.sort()
                self.expand_ratio_list.sort()
                self.width_mult_list.sort()
