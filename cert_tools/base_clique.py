import numpy as np

OVERLAP_ALL = False


class BaseClique(object):
    @staticmethod
    def get_overlap(cl, ck):
        return (
            set(cl.var_dict.keys()).intersection(ck.var_dict.keys()).difference({"h"})
        )

    def __init__(
        self,
        Q,
        A_list=[],
        b_list=[],
        left_slice_start=[],
        left_slice_end=[],
        right_slice_start=[],
        right_slice_end=[],
        var_dict=None,
        X=None,
    ):
        assert Q is not None or X is not None
        self.Q = Q

        self.A_list = A_list
        self.b_list = b_list
        self.left_slice_start = left_slice_start
        self.left_slice_end = left_slice_end
        self.right_slice_start = right_slice_start
        self.right_slice_end = right_slice_end

        if var_dict is not None:
            assert "h" in var_dict, "Each clique must has a h."
        self.var_dict = var_dict
        self.var_start_index = None
        self.X = X

        self.X_dim = Q.shape[0] if Q is not None else X.shape[0]

    def get_ranges(self, var_key: str):
        """Return the index range of var_key.

        :param var_key: name of overlapping  variable
        """
        if self.var_start_index is None:
            self.var_start_index = dict(
                zip(self.var_dict.keys(), np.cumsum([0] + list(self.var_dict.values())))
            )

        h_range = range(self.var_start_index["h"], self.var_start_index["h"] + 1)
        var_range = range(
            self.var_start_index[var_key],
            self.var_start_index[var_key] + self.var_dict[var_key],
        )
        if OVERLAP_ALL:
            return [[h_range, var_range]] + [[var_range, var_range]]
        else:
            return [[h_range, var_range]]

    def __repr__(self):
        vars_pretty = tuple(self.var_dict.keys()) if self.var_dict is not None else ()
        return f"clique var_dict={vars_pretty}"
