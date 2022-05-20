import torch.nn as nn


def get_net_info(net, input_shape=(3, 224, 224), measure_latency=None, print_info=True):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info["params"] = count_parameters(net) / 1e6

    # flops
    net_info["flops"] = count_net_flops(net, [1] + list(input_shape)) / 1e6

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split("#")
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(
            net, l_type, fast=False, input_shape=input_shape
        )
        net_info["%s latency" % l_type] = {"val": latency, "hist": measured_latency}

    if print_info:
        print(net)
        print("Total training params: %.2fM" % (net_info["params"]))
        print("Total FLOPs: %.2fM" % (net_info["flops"]))
        for l_type in latency_types:
            print(
                "Estimated %s latency: %.3fms"
                % (l_type, net_info["%s latency" % l_type]["val"])
            )

    return net_info