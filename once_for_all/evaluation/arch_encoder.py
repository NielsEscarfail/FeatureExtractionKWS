
class KWSNetArchEncoder:
    def __init__(
        self,
        ft_extr_params_list=None,
        width_mult_list=None,
        ks_list=None,
        depth_list=None,
        expand_list=None,
        base_depth_list=None,
    ):
        self.ft_extr_params_list = [(40, 40)] if ft_extr_params_list is None else ft_extr_params_list

        self.ks_list = [3, 5, 7] if ks_list is None else ks_list
        self.expand_list = [3, 2, 1] if expand_list is None else expand_list
        self.depth_list = [1, 2, 3, 4] if depth_list is None else depth_list
        self.width_mult_list = (
            [1.0] if width_mult_list is None else width_mult_list
        )

        self.base_depth_list = (
            [64, 64, 64, 64] if base_depth_list is None else base_depth_list
        )

        """" build info dict """
        self.n_dim = 0
        # resolution
        self.r_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r")
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
        if target == "r":
            target_dict = self.r_info
            target_dict["L"].append(self.n_dim)
            for ft_extr_params in self.ft_extr_params_list:
                target_dict["val2id"][ft_extr_params] = self.n_dim
                target_dict["id2val"][self.n_dim] = ft_extr_params
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "width_mult":
            target_dict = self.width_mult_info
            choices = list(range(len(self.width_mult_list)))
            for i in range(self.n_stage + 1):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for w in choices:
                    target_dict["val2id"][i][w] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = w
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)
        else:
            if target == "k":
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
        d, e, w, r = (
            arch_dict["d"],
            arch_dict["e"],
            arch_dict["w"],
            arch_dict["image_size"],
        )
        input_stem_skip = 1 if d[0] > 0 else 0
        d = d[1:]

        feature = np.zeros(self.n_dim)
        feature[self.r_info["val2id"][r]] = 1
        feature[self.input_stem_d_info["val2id"][input_stem_skip]] = 1
        for i in range(self.n_stage + 2):
            feature[self.width_mult_info["val2id"][i][w[i]]] = 1

        start_pt = 0
        for i, base_depth in enumerate(self.base_depth_list):
            depth = base_depth + d[i]
            for j in range(start_pt, start_pt + depth):
                feature[self.e_info["val2id"][j][e[j]]] = 1
            start_pt += max(self.depth_list) + base_depth

        return feature

    def arch2feature(self, arch_dict):
        ks, e, d, r = (
            arch_dict["ks"],
            arch_dict["e"],
            arch_dict["d"],
            arch_dict["image_size"],
        )

        feature = np.zeros(self.n_dim)
        for i in range(self.max_n_blocks):
            nowd = i % max(self.depth_list)
            stg = i // max(self.depth_list)
            if nowd < d[stg]:
                feature[self.k_info["val2id"][i][ks[i]]] = 1
                feature[self.e_info["val2id"][i][e[i]]] = 1
        feature[self.r_info["val2id"][r]] = 1
        return feature