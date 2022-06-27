

class KWSNetArchEncoder:
    SPACE_TYPE = "kwsnet"

    def __init__(
            self,
            image_size_list=None,
            ks_list=None,
            expand_list=None,
            depth_list=None,
            n_stage=None,
    ):
        self.image_size_list = [224] if image_size_list is None else image_size_list
        self.ks_list = [3, 5, 7] if ks_list is None else ks_list
        self.expand_list = (
            [2, 1, .5]
            if expand_list is None
            else [int(expand) for expand in expand_list]
        )
        self.depth_list = [1, 2, 3, 4] if depth_list is None else depth_list
        if n_stage is not None:
            self.n_stage = n_stage
        elif self.SPACE_TYPE == "kwsnet":
            self.n_stage = 4
        else:
            raise NotImplementedError

        # build info dict
        self.n_dim = 0
        self.r_info = dict(id2val={}, val2id={}, L=[], R=[])  # resolution
        self._build_info_dict(target="r")

        self.k_info = dict(id2val=[], val2id=[], L=[], R=[])  # ks
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])  # expand
        self._build_info_dict(target="k")
        self._build_info_dict(target="e")

    @property
    def max_n_blocks(self):
        if self.SPACE_TYPE == "kwsnet":
            return self.n_stage * max(self.depth_list)
        else:
            raise NotImplementedError

    def _build_info_dict(self, target):
        if target == "r":  # resolution
            target_dict = self.r_info
            target_dict["L"].append(self.n_dim)
            for img_size in self.image_size_list:
                target_dict["val2id"][img_size] = self.n_dim
                target_dict["id2val"][self.n_dim] = img_size
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        else:
            if target == "k":   # ks
                target_dict = self.k_info
                choices = self.ks_list
            elif target == "e":   # expand
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