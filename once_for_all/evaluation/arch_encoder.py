import random
import numpy as np


class KWSNetArchEncoder:
    def __init__(
            self,
            ft_extr_params1_list=None,
            ft_extr_params2_list=None,
            width_mult_list=None,
            ks_list=None,
            depth_list=None,
            expand_list=None,
            base_depth_list=None,
    ):
        self.ft_extr_params1_list = [10, 20, 30, 40] if ft_extr_params1_list is None else ft_extr_params1_list
        self.ft_extr_params2_list = [30, 40, 50] if ft_extr_params2_list is None else ft_extr_params2_list
        self.width_mult_list = [1.0] if width_mult_list is None else width_mult_list
        self.ks_list = [3, 5, 7] if ks_list is None else ks_list
        self.expand_list = [3, 2, 1] if expand_list is None else expand_list
        self.depth_list = [1, 2, 3, 4] if depth_list is None else depth_list
        self.base_depth_list = [64, 64, 64, 64] if base_depth_list is None else base_depth_list

        """" build info dict """
        self.n_dim = 0
        # resolution (dim1)
        self.r1_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r1")
        # resolution (dim2)
        self.r2_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r2")
        # width_mult
        self.width_mult_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="width_mult")
        # kernel size
        self.k_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="k")
        # expand ratio
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="e")

    @property
    def n_stage(self):
        return len(self.base_depth_list)

    @property
    def max_n_blocks(self):
        return sum(self.base_depth_list) + self.n_stage * max(self.depth_list)

    def _build_info_dict(self, target):
        if target == "r1":
            target_dict = self.r1_info
            target_dict["L"].append(self.n_dim)
            for ft_extr_params1 in self.ft_extr_params1_list:
                target_dict["val2id"][ft_extr_params1] = self.n_dim
                target_dict["id2val"][self.n_dim] = ft_extr_params1
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "r2":
            target_dict = self.r2_info
            target_dict["L"].append(self.n_dim)
            for ft_extr_params2 in self.ft_extr_params1_list:
                target_dict["val2id"][ft_extr_params2] = self.n_dim
                target_dict["id2val"][self.n_dim] = ft_extr_params2
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        else:
            if target == "width_mult":
                target_dict = self.width_mult_info
                choices = list(range(len(self.width_mult_list)))
            elif target == "k":
                target_dict = self.k_info
                choices = self.ks_list
            elif target == "e":
                target_dict = self.e_info
                choices = self.expand_list
            else:
                raise NotImplementedError
            for i in range(self.max_n_blocks):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for k in choices:
                    target_dict["val2id"][i][k] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = k
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict):
        w, ks, e, d, r1, r2 = (
            arch_dict["width_mult"],
            arch_dict["ks"],
            arch_dict["e"],
            arch_dict["d"],
            arch_dict["ft_extr_params1"],
            arch_dict["ft_extr_params2"]
        )
        feature = np.zeros(self.n_dim)
        feature[self.r1_info["val2id"][r1]] = 1
        feature[self.r1_info["val2id"][r2]] = 1

        for i in range(self.n_stage + 1):
            feature[self.width_mult_info["val2id"][i][w[i]]] = 1

        start_pt = 0
        for i, base_depth in enumerate(self.base_depth_list):
            depth = base_depth + d[i]
            for j in range(start_pt, start_pt + depth):
                feature[self.e_info["val2id"][j][e[j]]] = 1
                feature[self.k_info["val2id"][j][ks[j]]] = 1
            start_pt += max(self.depth_list) + base_depth

        return feature

    def feature2arch(self, feature):
        # Feature extraction parameters
        ft_extr_params1 = self.r1_info["id2val"][
            int(np.argmax(feature[self.r1_info["L"][0]: self.r1_info["R"][0]]))
            + self.r1_info["L"][0]
            ]
        ft_extr_params2 = self.r2_info["id2val"][
            int(np.argmax(feature[self.r2_info["L"][0]: self.r2_info["R"][0]]))
            + self.r2_info["L"][0]
            ]
        assert ft_extr_params1 in self.ft_extr_params1_list
        assert ft_extr_params2 in self.ft_extr_params2_list

        arch_dict = {"w": [], "ks": [], "e": [], "d": [],
                     "ft_extr_params1": ft_extr_params1, "ft_extr_params2": ft_extr_params2}

        for i in range(self.n_stage + 1):
            # width
            arch_dict["w"].append(
                self.width_mult_info["id2val"][i][int(
                    np.argmax(feature[self.width_mult_info["L"][i]: self.width_mult_info["R"][i]])
                ) + self.width_mult_info["L"][i]]
            )

        d = 0
        for i in range(self.max_n_blocks):
            skip = True
            for j in range(self.k_info["L"][i], self.k_info["R"][i]):
                if feature[j] == 1:
                    # kernel size
                    arch_dict["ks"].append(self.k_info["id2val"][i][j])
                    skip = False
                    break

            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    # expand ratio
                    arch_dict["e"].append(self.e_info["id2val"][i][j])
                    assert not skip
                    skip = False
                    break

            if skip:
                arch_dict["e"].append(0)
                arch_dict["ks"].append(0)
            else:
                d += 1

            if (i + 1) % max(self.depth_list) == 0 or (i + 1) == self.max_n_blocks:
                # depth
                arch_dict["d"].append(d)
                d = 0

        return arch_dict

    def random_sample_arch(self):
        return {
            "w": random.choices(
                list(range(len(self.width_mult_list))), k=self.n_stage + 1
            ),
            "ks": random.choices(self.ks_list, k=self.max_n_blocks),
            "e": random.choices(self.expand_list, k=self.max_n_blocks),
            "d": random.choices(self.depth_list, k=self.n_stage),
            "ft_extr_params1": random.choice(self.ft_extr_params1_list),
            "ft_extr_params2": random.choice(self.ft_extr_params2_list),
        }

    def mutate_resolution(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["ft_extr_params1"] = random.choice(self.ft_extr_params1_list)
            arch_dict["ft_extr_params2"] = random.choice(self.ft_extr_params2_list)
        return arch_dict

    def mutate_arch(self, arch_dict, mutate_prob):
        # width_mult
        for i in range(len(arch_dict["w"])):
            if random.random() < mutate_prob:
                arch_dict["w"][i] = random.choice(
                    list(range(len(self.width_mult_list)))
                )

        # kernel size
        for i in range(1, len(arch_dict["ks"])):
            if random.random() < mutate_prob:
                arch_dict["ks"][i] = random.choice(self.ks_list)

        # depth
        for i in range(1, len(arch_dict["d"])):
            if random.random() < mutate_prob:
                arch_dict["d"][i] = random.choice(self.depth_list)

        # expand ratio
        for i in range(len(arch_dict["e"])):
            if random.random() < mutate_prob:
                arch_dict["e"][i] = random.choice(self.expand_list)
