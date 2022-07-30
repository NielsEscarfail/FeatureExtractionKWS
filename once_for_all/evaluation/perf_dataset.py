# Created by: Niels Escarfail, ETH (nescarfail@student.ethz.ch)


import json
import os

from tqdm import tqdm

from utils import list_mean, get_net_info


class PerformanceDataset:
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    @staticmethod
    def net_setting2id(net_setting):
        return json.dumps(net_setting)

    @staticmethod
    def net_id2setting(self, net_id):
        return json.loads(net_id)

    def net_setting_in_df(self, net_setting, df):
        equal_condition = df['w'] == net_setting['w'] & \
                          df['ks'] == net_setting['ks'] & \
                          df['d'] == net_setting['d'] & \
                          df['e'] == net_setting['e']

        return equal_condition

    @property
    def net_id_path(self):
        return os.path.join(self.path, "net_id.dict")

    @property
    def perf_src_folder(self):
        return os.path.join(self.path, "src")

    @property
    def perf_dict_path(self):
        return os.path.join(self.path, "perf.dict")

    def load_net_id_list(self, ofa_net, n_arch):
        # Load a net_id_list if the net_id_list already gathered is long enough
        if os.path.isfile(self.net_id_path):
            net_id_list = json.load(open(self.net_id_path))
            if len(net_id_list) > n_arch:
                return net_id_list

        # Reload completely otherwise
        net_id_list = set()
        while len(net_id_list) < n_arch:
            net_setting = ofa_net.sample_active_subnet()
            net_id = self.net_setting2id(net_setting)
            net_id_list.add(net_id)
        net_id_list = list(net_id_list)
        net_id_list.sort()
        # Save sampled net_id_list
        json.dump(net_id_list, open(self.net_id_path, "w"), indent=4)
        return net_id_list

    def build_dataset(self, run_manager, ofa_net, n_arch=1000, ft_extr_params_list=None, measure_latency=None):
        """
        Samples network architectures and saves the :
        - network configuration:
        - top1: top 1 accuracy
        - params: number of network parameters in M (/1e6)
        - flops: flops in M (/1e6)
        - latencies: (val, hist) latency and measured latency latencies for selected types, in ms
        CAN BE ADDED IF CONFIGURATION ITSELF IS NOT VIABLE / TOO LARGE
        - net_encoding: Encoding which can be used to recover the network
        """
        print("Using json")
        net_id_list = self.load_net_id_list(ofa_net, n_arch)

        print("Evaluating %i sub-networks for %i ft_extr_params: " % (len(net_id_list), len(ft_extr_params_list)))

        with tqdm(
                total=len(net_id_list) * len(ft_extr_params_list), desc="Building Performance Dataset"
        ) as t:
            for ft_extr_params in ft_extr_params_list:
                print(ft_extr_params)
                # load val dataset into memory
                val_dataset = []
                run_manager.run_config.data_provider.assign_active_ft_extr_params(ft_extr_params)
                for images, labels in run_manager.run_config.valid_loader:
                    val_dataset.append((images, labels))
                print("loaded dataset")

                # save path
                os.makedirs(self.perf_src_folder, exist_ok=True)
                perf_save_path = os.path.join(self.perf_src_folder, "%s.dict" % str(list(ft_extr_params)))
                perf_dict = {}

                # load existing performance dict
                if os.path.isfile(perf_save_path):
                    existing_perf_dict = json.load(open(perf_save_path, "r"))
                    print("Loaded existing performance dict of length: ", len(existing_perf_dict))
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

                    # Set sampled subnet
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

                    # Gather performance results
                    loss, (top1, top5) = run_manager.validate(
                        run_str=net_setting_str,
                        net=ofa_net,
                        data_loader=val_dataset,
                        no_logs=True,
                    )
                    data_shape = val_dataset[0][0].shape[1:]

                    # Gets n_params, flops,
                    active_subnet = ofa_net.get_active_subnet()
                    net_info = get_net_info(active_subnet,
                                            input_shape=data_shape,
                                            measure_latency=measure_latency,
                                            print_info=False)
                    info_val = {
                        "ft_extr_params": ft_extr_params,
                        "data_shape": data_shape,
                        "top1": top1,
                        "net_info": net_info,
                    }

                    t.set_postfix(
                        {
                            "net_id": net_id,
                            "acc": top1,
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

    def load_dataset(self, ft_extr_params):
        perf_save_path = os.path.join(self.perf_src_folder, "%s.dict" % str(list(ft_extr_params)))
        # load data
        data = json.load(open(perf_save_path))
        return data
