import numpy as np
from poly_matrix import PolyMatrix

OVERLAP_ALL = False


class BaseClique(object):
    """Base class used to represent a clique in the aggregate sparsity
    graph (ASG). Used to store useful information about the clique, namely
    variable ordering and sizes stored in var_sizes attribute.
    Also stores the distributed cost and constraint values.

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """

    @staticmethod
    def get_overlap(cl, ck, h="h"):
        "This function is now redundant"
        return set(cl.var_dict.keys()).intersection(ck.var_dict.keys()).difference(h)

    def __init__(
        self,
        index,
        var_sizes,
        Q: PolyMatrix = PolyMatrix(),
        A_list=[],
        b_list=[],
    ):
        self.index = index
        if var_sizes is not None:
            assert "h" in var_sizes, f"Each clique must have a homogenizing variable"
        self.var_sizes = var_sizes
        self.var_inds, self.size = self._get_start_indices()
        # Assign cost and constraints if provided
        self.Q = Q
        self.A_list = A_list
        self.b_list = b_list

    def _get_start_indices(self):
        "Loop through variable sizes and get starting indices"
        index = 0
        var_inds = {}
        for varname in self.var_sizes.keys():
            var_inds[varname] = index
            index += self.var_sizes[varname]
        size = index
        return var_inds, size

    def __repr__(self):
        vars_pretty = tuple(self.var_sizes.keys()) if self.var_sizes is not None else ()
        return f"clique var_sizes={vars_pretty}"

    def get_slices(self, mat, var_list):
        """get slices according to variable ordering"""
        slices = []
        for varname in var_list:
            start = self.var_inds[varname]
            end = self.var_inds[varname] + self.var_sizes[varname]
            slices.append(np.array(range(start, end)))
        inds = np.hstack(slices)
        return mat[np.ix_(inds, inds)]
