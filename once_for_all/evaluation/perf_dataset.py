import json
import os

import pandas as pd
from tqdm import tqdm

from utils import list_mean, count_parameters, count_net_flops, get_net_info


def net_setting2id(net_setting):
    return json.dumps(net_setting)


class PerformanceDataset:
    def __init__(self, path, use_csv):
        self.path = path
        self.use_csv = use_csv
        os.makedirs(self.path, exist_ok=True)

    def net_id2setting(self, net_id):
        if self.use_csv:
            return net_id.to_dict()
        else:
            return json.loads(net_id)

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
        if self.use_csv:
            # Load a net_id_list
            if os.path.isfile(self.net_id_path):
                net_id_list = pd.read_csv(self.net_id_path)
                print("loaded net id list : ", net_id_list)
            else:
                net_id_list = []
                while len(net_id_list) < n_arch:
                    net_setting = ofa_net.sample_active_subnet()
                    net_id_list.append(net_setting)

                print("net_id_list before save : ", net_id_list)
                net_id_list = pd.DataFrame.from_dict(net_id_list)
                print("net_id_list df : ", net_id_list)
                net_id_list.to_csv(self.net_id_path, index=False)
            # ft_extr_type = "mfcc" if ft_extr_type is None else ft_extr_type
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
                    perf_df = None
                    # load existing performance dict
                    if os.path.isfile(perf_save_path):
                        existing_perf_df = pd.read_csv(perf_save_path)
                    else:
                        existing_perf_df = {}

                    print("net_id_list : ", net_id_list)
                    print("type ", type(net_id_list))
                    print("cols : ", net_id_list.columns)
                    for index, net_id in net_id_list.iterrows():
                        print("net_id : ", net_id)
                        print(type(net_id))
                        net_setting = self.net_id2setting(net_id)
                        key = net_setting2id({**net_setting, "ft_extr_params": ft_extr_params})
                        if key in existing_perf_df:  # If setting already logged, don't test
                            perf_df[key] = existing_perf_df[key]
                            t.set_postfix(
                                {
                                    "net_id": net_id,
                                    "ft_extr_params": ft_extr_params,
                                    "info_val": perf_df[key],
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
                        loss, (top1, top5) = run_manager.validate(
                            run_str=net_setting_str,
                            net=ofa_net,
                            data_loader=val_dataset,
                            no_logs=True,
                        )
                        data_shape = val_dataset[0][0].shape[1:]
                        print("testing: data_shape: ", data_shape)
                        info_val = {
                            "ft_extr_params": ft_extr_params,
                            "data_shape": data_shape,
                            "top1": top1,
                            "net_info": get_net_info(ofa_net,
                                                     input_shape=data_shape,
                                                     measure_latency="gpu4#cpu",
                                                     print_info=False),
                            # Gets n_params, flops, latency for gpu4, cpu
                        }
                        t.set_postfix(
                            {
                                "net_id": net_id,
                                "ft_extr_params": ft_extr_params,
                                "info_val": info_val,
                            }
                        )
                        t.update()

                        perf_df.update({key: info_val})  # Save accuracy, net_info
                        perf_df.to_csv(perf_save_path)

        else:  # Use json
            # Load a net_id_list
            if os.path.isfile(self.net_id_path):
                net_id_list = json.load(open(self.net_id_path))
            else:
                net_id_list = set()
                while len(net_id_list) < n_arch:
                    net_setting = ofa_net.sample_active_subnet()
                    net_id = net_setting2id(net_setting)
                    net_id_list.add(net_id)
                net_id_list = list(net_id_list)
                net_id_list.sort()
                # Save sampled net_id_list
                json.dump(net_id_list, open(self.net_id_path, "w"), indent=4)

        # ft_extr_type = "mfcc" if ft_extr_type is None else ft_extr_type
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
                perf_save_path = os.path.join(self.perf_src_folder, "%s.dict" % str(list(ft_extr_params)))
                perf_dict = {}
                # load existing performance dict
                if os.path.isfile(perf_save_path):
                    existing_perf_dict = json.load(open(perf_save_path, "r"))
                else:
                    existing_perf_dict = {}

                for net_id in net_id_list:
                    net_setting = self.net_id2setting(net_id)
                    key = net_setting2id({**net_setting, "ft_extr_params": ft_extr_params})
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
                    loss, (top1, top5) = run_manager.validate(
                        run_str=net_setting_str,
                        net=ofa_net,
                        data_loader=val_dataset,
                        no_logs=True,
                    )
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

    def load_dataset(self):
        # load data
        return json.load(open(self.perf_dict_path))  # eval/perf.dict
