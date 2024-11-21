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

    def __init__(
        self,
        index,
        var_sizes,
        parent,
        separator,
        Q: PolyMatrix = PolyMatrix(),
        A_list=[],
        b_list=[],
    ):
        self.index = index
        # if var_sizes is not None:
        #     assert "h" in var_sizes, f"Each clique must have a homogenizing variable"
        self.var_sizes = var_sizes
        self.var_list = list(self.var_sizes.keys())
        self.var_inds, self.size = self._get_start_indices()
        # Store clique tree information
        for key in separator:
            assert (
                key in self.var_sizes.keys()
            ), f"separator element {key} not contained in clique"
        self.separator = separator  # separator set between this clique and its parent
        self.residual = self.var_list.copy()
        for varname in separator:
            self.residual.remove(varname)
        self.parent = parent  # index of the parent clique
        self.children = set()  # set of children of this clique in the clique tree
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

    def add_children(self, children: list = []):
        self.children.add(children)

    def _get_indices(self, var_list):
        """get the indices corresponding to a list of variable keys

        Args:
            var_list: variable key or list of variable keys

        Returns:
            _type_: _description_
        """
        if type(var_list) is not list:
            if type(var_list) is str:
                var_list = [var_list]
            else:
                var_list = list(var_list)
        # Get index slices for the rows
        slices = []
        for varname in var_list:
            start = self.var_inds[varname]
            end = self.var_inds[varname] + self.var_sizes[varname]
            slices.append(np.array(range(start, end)))
        inds = np.hstack(slices)
        return inds

    def get_slices(self, mat, var_list_row, var_list_col=[]):
        """Get slices according to prescribed variable ordering.
        If one list provided then slices are assumed to be symmetric. If two lists are provided, they are interpreted as the row and column lists, respectively.
        """
        # Get index slices for the rows
        inds1 = self._get_indices(var_list_row)
        # Get index slices for the columns
        if len(var_list_col) > 0:
            inds2 = self._get_indices(var_list_col)
        else:
            # If not defined use the same list as rows
            inds2 = inds1

        return mat[np.ix_(inds1, inds2)]
