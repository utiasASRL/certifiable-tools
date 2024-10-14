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

    def __repr__(self):
        vars_pretty = tuple(self.var_sizes.keys()) if self.var_sizes is not None else ()
        return f"clique var_sizes={vars_pretty}"

    def _get_start_indices(self):
        "Loop through variable sizes and get starting indices"
        index = 0
        var_inds = {}
        for varname in self.var_sizes.keys():
            var_inds[varname] = index
            index += self.var_sizes[varname]
        size = index
        return var_inds, size

    def get_slices(self, mat, var_list_row, var_list_col=[]):
        """Get slices according to prescribed variable ordering.
        If one list provided then slices are assumed to be symmetric. If two lists are provided, they are interpreted as the row and column lists, respectively.
        """
        slices = []
        # Get index slices for the rows
        for varname in var_list_row:
            start = self.var_inds[varname]
            end = self.var_inds[varname] + self.var_sizes[varname]
            slices.append(np.array(range(start, end)))
        inds1 = np.hstack(slices)
        # Get index slices for the columns
        if len(var_list_col) > 0:
            slices = []
            for varname in var_list_col:
                start = self.var_inds[varname]
                end = self.var_inds[varname] + self.var_sizes[varname]
                slices.append(np.array(range(start, end)))
            inds2 = np.hstack(slices)
        else:
            # If not defined use the same list as rows
            inds2 = inds1

        return mat[np.ix_(inds1, inds2)]
