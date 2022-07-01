import json
import os

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from utils import list_mean, get_net_info, AverageMeter


class PerformanceDataset:
    def __init__(self, path, use_csv):
        self.path = path
        self.use_csv = use_csv
        os.makedirs(self.path, exist_ok=True)

    def net_setting2id(self, net_setting):
        if self.use_csv:
            return json.dumps(net_setting)
        else:
            return json.dumps(net_setting)

    def net_id2setting(self, net_id):
        if self.use_csv:
            return net_id
        else:
            return json.loads(net_id)

    def net_setting_in_df(self, net_setting, df):
        equal_condition = df['w'] == net_setting['w'] & \
                          df['ks'] == net_setting['ks'] & \
                          df['d'] == net_setting['d'] & \
                          df['e'] == net_setting['e']

        return equal_condition

    @property
    def net_id_path(self):
        return os.path.join(self.path, "net_id.csv") if self.use_csv else os.path.join(self.path, "net_id.dict")

    @property
    def perf_src_folder(self):
        return os.path.join(self.path, "src")

    @property
    def perf_dict_path(self):
        return os.path.join(self.path, "perf.csv") if self.use_csv else os.path.join(self.path, "perf.dict")

    def build_dataset(self, run_manager, ofa_net, n_arch=1000, ft_extr_params_list=None):
        """
        Samples network architectures and saves the :
        - network configuration:
        - top1: top 1 accuracy
        - params: number of network parameters in M (/1e6)
        - flops: flops in M (/1e6)
        - latencies: (val, hist) latency and measured latency latencies for selected types, in ms
        MIGHT BE ADDED IF CONFIGURATION ITSELF IS NOT VIABLE / TOO LARGE
        - net_encoding: Encoding which can be used to recover the network
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        ofa_net.to(device)
        if self.use_csv:
            print("Using csv")
            # Load a net_id_list
            if os.path.isfile(self.net_id_path):
                net_id_df = pd.read_csv(self.net_id_path, converters={"w": lambda x: x.strip("[]").split(", "),
                                                                      "ks": lambda x: x.strip("[]").split(", "),
                                                                      "d": lambda x: x.strip("[]").split(", "),
                                                                      "e": lambda x: x.strip("[]").split(", ")
                                                                      })
                print("Loaded : ", net_id_df)
                print("\n\n")
                net_id_list = net_id_df.values.tolist()
            else:
                net_id_list = []
                while len(net_id_list) < n_arch:
                    net_setting = ofa_net.sample_active_subnet()
                    net_id_list.append(net_setting)

                net_id_df = pd.DataFrame(net_id_list)
                net_id_df.to_csv(self.net_id_path, index=False)

            ft_extr_params_list = (
                [(40, 30), (40, 40), (40, 50)] if ft_extr_params_list is None else ft_extr_params_list
            )

            with tqdm(
                    total=len(net_id_list) * len(ft_extr_params_list), desc="Building Performance Dataset"
            ) as t:
                for ft_extr_params in ft_extr_params_list:
                    # load val dataset into memory
                    val_dataset = []
                    run_manager.run_config.data_provider.assign_active_ft_extr_params(ft_extr_params)
                    for images, labels in run_manager.run_config.valid_loader:
                        val_dataset.append((images, labels))
                    # save path
                    os.makedirs(self.perf_src_folder, exist_ok=True)

                    perf_save_path = os.path.join(self.perf_src_folder, "%s.csv" % str(list(ft_extr_params)))
                    perf_list = []

                    # load existing performance dict
                    if os.path.isfile(perf_save_path):
                        existing_perf_df = pd.read_csv(perf_save_path,
                                                       converters={"w": lambda x: x.strip("[]").split(", "),
                                                                   "ks": lambda x: x.strip("[]").split(", "),
                                                                   "d": lambda x: x.strip("[]").split(", "),
                                                                   "e": lambda x: x.strip("[]").split(", ")
                                                                   })
                        print("Loaded existing performance: ", existing_perf_df.head(2))
                        existing_perf_list = existing_perf_df.values.tolist()
                    else:
                        existing_perf_df = None
                        existing_perf_list = []

                    for net_id in net_id_list:
                        # net_setting = self.net_id2setting(net_id)
                        net_setting = net_id
                        print("net setting ", net_setting)
                        print("type net setting ", type(net_setting))
                        print("type wi ", type(net_setting['w']))
                        key = self.net_setting2id({**net_setting, "ft_extr_params": ft_extr_params})

                        """Add to already loaded performance if it exists"""
                        if existing_perf_df is not None:
                            already_evaluated = self.net_setting_in_df(net_setting, existing_perf_df)
                            if already_evaluated:  # If setting already logged, don't test
                                # perf_df.append(existing_perf_df[key])
                                perf_list = perf_list.append(existing_perf_df[already_evaluated])
                                t.set_postfix(
                                    {
                                        "net_id": net_id,
                                        "ft_extr_params": ft_extr_params,
                                        "info_val": existing_perf_df[already_evaluated],
                                        "status": "loading",
                                    }
                                )
                                t.update()
                                continue

                        """Set subnet and record performance"""
                        ofa_net.set_active_subnet(**net_setting)
                        run_manager.reset_running_statistics(ofa_net, )
                        net_setting_str = ",".join(
                            [
                                "%s_%s"
                                % (
                                    key,
                                    "%.1f" % list_mean(val)
                                    if isinstance(val, list)
                                    else val,
                                )
                                for key, val in net_setting.items()
                            ]
                        )
                        loss, (top1, top5) = run_manager.validate(
                            run_str=net_setting_str,
                            net=ofa_net,
                            data_loader=val_dataset,
                            no_logs=True,
                        )
                        """Create net_info/perf dict and append to net_perf_list"""
                        data_shape = val_dataset[0][0].shape[1:]
                        # Gets n_params, flops, latency for gpu4, cpu
                        net_info = get_net_info(ofa_net,
                                                input_shape=data_shape,
                                                measure_latency="gpu4#cpu",
                                                print_info=False)

                        norm_net_info = pd.json_normalize(net_info, sep='_')
                        norm_net_info["ft_extr_params_1"] = ft_extr_params[0]
                        norm_net_info["ft_extr_params_2"] = ft_extr_params[1]
                        norm_net_info["data_shape"] = str(data_shape)
                        norm_net_info["top1"] = top1
                        norm_net_info['key'] = key

                        print("NORM NET INFO ")
                        print(norm_net_info)
                        print()

                        norm_net_info.update({'w': net_setting['w']})
                        norm_net_info.update({'ks': net_setting['ks']})
                        norm_net_info.update({'e': net_setting['e']})
                        norm_net_info.update({'d': net_setting['d']})
                        """norm_net_info['ks'] = net_setting['ks']
                        norm_net_info['e'] = net_setting['e']
                        norm_net_info['d'] = net_setting['d']"""

                        # Display
                        info_val = {  # For display purposes for now
                            "top1": top1,
                        }
                        t.set_postfix(
                            {
                                "net_id": net_id,
                                "ft_extr_params": ft_extr_params,
                                "info_val": info_val,
                            }
                        )
                        t.update()

                        """Save the performance data"""
                        perf_list.append(norm_net_info)

                        """ if perf_df is None:
                            # print("perf df is none before : ", perf_df)
                            perf_df = pd.DataFrame(data=norm_net_info)
                            perf_df.set_index(['w', 'ks', 'e', 'd'])
                        else:
                            # info = pd.DataFrame(data=norm_net_info)
                            # info.set_index(['w', 'ks', 'e', 'd'])
                            perf_df = perf_df.append(info)
                            print("perf df not none afterupdate : ", perf_df)"""

                        print("pref df : ", perf_list)

                    perf_df = pd.DataFrame(data=perf_list)
                    perf_df.set_index(['w', 'ks', 'e', 'd', 'ft_extr_params1', 'ft_extr_params2'])
                    perf_df.to_csv(perf_save_path)
                    print("Saved to csv: ")
                    print(perf_df)
                    print()

        else:  # Use json
            print("Using json")
            # Load a net_id_list
            if os.path.isfile(self.net_id_path):
                net_id_list = json.load(open(self.net_id_path))
            else:
                net_id_list = set()
                while len(net_id_list) < n_arch:
                    net_setting = ofa_net.sample_active_subnet()
                    net_id = self.net_setting2id(net_setting)
                    net_id_list.add(net_id)
                net_id_list = list(net_id_list)
                net_id_list.sort()
                # Save sampled net_id_list
                json.dump(net_id_list, open(self.net_id_path, "w"), indent=4)

            ft_extr_params_list = ([(40, 30), (40, 40), (40, 50)] if ft_extr_params_list is None else ft_extr_params_list)

            with tqdm(
                total=len(net_id_list) * len(ft_extr_params_list), desc="Building Performance Dataset"
            ) as t:
                for ft_extr_params in ft_extr_params_list:
                    # load val dataset into memory
                    val_dataset = []
                    run_manager.run_config.data_provider.assign_active_ft_extr_params(ft_extr_params)
                    for images, labels in run_manager.run_config.valid_loader:
                        val_dataset.append((images, labels))
                    # save path
                    os.makedirs(self.perf_src_folder, exist_ok=True)
                    perf_save_path = os.path.join(self.perf_src_folder, "%s.dict" % str(list(ft_extr_params)))
                    perf_dict = {}
                    # load existing performance dict
                    if os.path.isfile(perf_save_path):
                        existing_perf_dict = json.load(open(perf_save_path, "r"))
                    else:
                        existing_perf_dict = {}

                    for net_id in net_id_list:
                        net_setting = self.net_id2setting(net_id)
                        key = self.net_setting2id({**net_setting, "ft_extr_params": ft_extr_params})
                        if key in existing_perf_dict:  # If setting already logged, don't test
                            perf_dict[key] = existing_perf_dict[key]
                            t.set_postfix(
                                {
                                    "net_id": net_id,
                                    "ft_extr_params": ft_extr_params,
                                    "info_val": perf_dict[key],
                                    "status": "loading",
                                }
                            )
                            t.update()
                            continue
                        ofa_net.set_active_subnet(**net_setting)
                        run_manager.reset_running_statistics(ofa_net)
                        net_setting_str = ",".join(
                            [
                                "%s_%s"
                                % (
                                    key,
                                    "%.1f" % list_mean(val)
                                    if isinstance(val, list)
                                    else val,
                                )
                                for key, val in net_setting.items()
                            ]
                        )
                        # get accuracy
                        ofa_net.eval()
                        losses = AverageMeter()
                        metric_dict = run_manager.get_metric_dict()
                        with torch.no_grad():
                            with tqdm(
                                    total=len(val_dataset),
                                    desc="Validate net {}".format(net_setting_str),
                                    disable=False,
                            ) as t2:
                                for i, (images, labels) in enumerate(val_dataset):
                                    images, labels = images.to(device), labels.to(device)
                                    # compute output
                                    output = ofa_net(images)
                                    loss = run_manager.test_criterion(output, labels)
                                    # measure accuracy and record loss
                                    run_manager.update_metric(metric_dict, output, labels)

                                    losses.update(loss.item(), images.size(0))

                        loss = losses.avg
                        (top1, top5) = run_manager.get_metric_vals(metric_dict)

                        """loss, (top1, top5) = run_manager.validate(
                            run_str=net_setting_str,
                            net=ofa_net,
                            data_loader=val_dataset,
                            no_logs=True,
                        )"""
                        data_shape = val_dataset[0][0].shape[1:]
                        info_val = {
                            "ft_extr_params": ft_extr_params,
                            "data_shape": data_shape,
                            "top1": top1,
                            "net_info": get_net_info(ofa_net,
                                                     input_shape=data_shape,
                                                     measure_latency="gpu4#cpu",
                                                     print_info=False),  # Gets n_params, flops, latency for gpu4, cpu
                        }
                        t.set_postfix(
                            {
                                "net_id": net_id,
                                "ft_extr_params": ft_extr_params,
                                "info_val": info_val,
                            }
                        )
                        t.update()

                        perf_dict.update({key: info_val})  # Save accuracy, net_info
                        json.dump(perf_dict, open(perf_save_path, "w"), indent=4)

    def merge_acc_dataset(self, ft_extr_params_list=None):
        # load existing data
        merged_perf_dict = {}
        for fname in os.listdir(self.perf_src_folder):
            if ".dict" not in fname:
                continue
            ft_extr_params = int(fname.split(".dict")[0])
            if ft_extr_params_list is not None and ft_extr_params not in ft_extr_params_list:
                print("Skip ", fname)
                continue
            full_path = os.path.join(self.perf_src_folder, fname)
            partial_perf_dict = json.load(open(full_path))
            merged_perf_dict.update(partial_perf_dict)
            print("loaded %s" % full_path)
        json.dump(merged_perf_dict, open(self.perf_dict_path, "w"), indent=4)
        return merged_perf_dict

    def json_to_csv_dataset(self):
        # load data
        data = json.load(open(self.perf_dict_path))  # eval/perf.dict
        df = pd.DataFrame(data)
