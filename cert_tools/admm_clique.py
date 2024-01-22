import itertools

import cvxpy as cp
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from cert_tools.base_clique import BaseClique

CONSTRAIN_ALL_OVERLAP = False


def initialize_overlap(clique_list):
    for k, clique in enumerate(clique_list):
        assert isinstance(clique, ADMMClique)

        # below are for solving with fusion API
        clique.generate_overlap_slices(N=len(clique_list))
        clique.generate_F(N=len(clique_list))

        # below are for solving with cvxpy
        left = clique_list[k - 1].X_var if k > 0 else None
        right = clique_list[k + 1].X_var if k < len(clique_list) - 1 else None
        clique.g = clique.generate_g(left=left, right=right)
        assert clique.F.shape[0] == clique.g.shape[0]


class ADMMClique(BaseClique):
    def __init__(
        self,
        Q,
        A_list: list,
        b_list: list,
        var_dict: dict,
        index,
        X: np.ndarray = None,
        N: int = 0,
        hom="l",
        x_dim: int = 0,
    ):
        super().__init__(
            Q=Q,
            A_list=A_list,
            b_list=b_list,
            var_dict=var_dict,
            X=X,
            index=index,
            hom=hom,
        )
        self.x_dim = x_dim
        self.constrain_all = CONSTRAIN_ALL_OVERLAP
        if self.constrain_all:
            self.num_overlap = x_dim**2 + x_dim
        else:
            self.num_overlap = x_dim
        self.status = 0

        self.X_var = cp.Variable((self.X_dim, self.X_dim), PSD=True)
        self.X_new = None
        self.z_new = None
        self.z_prev = None

        self.g = None

        assert N > 0, "must give total number of nodes N"
        self.F = self.generate_F(N=N)
        self.E = self.get_E(N=N)

    def get_E(self, N):
        """Create the indexing matrix that picks the k-th out of N cliques.

        For our example, this matrix is of the form:

             k-th block
                  |
                  v
         | 1     0 0     |
         | 0 ... 1 0 ... |
         | 0 ... 0 1 ... |

        """
        return sp.coo_matrix(
            (
                np.ones(self.X_dim),
                [
                    range(self.X_dim),
                    [0]
                    + list(
                        range(
                            1 + self.index * self.x_dim,
                            1 + self.index * self.x_dim + 2 * self.x_dim,
                        )
                    ),
                ],
            ),
            shape=(self.X_dim, 1 + N * self.x_dim),
        )

    def get_B_list_right(self):
        B_list = []
        B = np.zeros((self.X_dim, self.X_dim))
        B[0, 0] = 1.0
        # B_list.append(B)
        for i, j in itertools.combinations_with_replacement(range(self.x_dim), 2):
            B = np.zeros((self.X_dim, self.X_dim))
            B[1 + self.x_dim + i, 1 + self.x_dim + j] = 1.0
            B[1 + self.x_dim + j, 1 + self.x_dim + i] = 1.0
            B_list.append(B)
        for i in range(self.x_dim):
            B = np.zeros((self.X_dim, self.X_dim))
            B[0, 1 + self.x_dim + i] = 1.0
            B[1 + self.x_dim + i, 0] = 1.0
            B_list.append(B)
        return B_list

    def get_B_list_left(self):
        B_list = []
        B = np.zeros((self.X_dim, self.X_dim))
        B[0, 0] = -1.0
        # B_list.append(B)
        for i, j in itertools.combinations_with_replacement(range(self.x_dim), 2):
            B = np.zeros((self.X_dim, self.X_dim))
            B[1 + i, 1 + j] = -1.0
            B[1 + j, 1 + i] = -1.0
            B_list.append(B)
        for i in range(self.x_dim):
            B = np.zeros((self.X_dim, self.X_dim))
            B[0, 1 + i] = -1.0
            B[1 + i, 0] = -1.0
            B_list.append(B)
        return B_list

    def generate_g(self, left=None, right=None):
        """Generate vector for overlap constraints: F @ X.flatten() - g = 0"""
        vec_all = None
        if left is not None:
            vec_all = self.F_right @ left.flatten()
        if right is not None:
            vec_new = self.F_left @ right.flatten()
            if isinstance(right, np.ndarray):
                vec_all = (
                    np.hstack([vec_all, vec_new]) if vec_all is not None else vec_new
                )
            elif isinstance(right, cp.Variable):
                vec_all = (
                    cp.hstack([vec_all, vec_new]) if vec_all is not None else vec_new
                )
            else:
                raise TypeError("unexpected type")
        return vec_all

    def generate_F(self, N):
        """Generate matrix for overlap constraints: F @ X.flatten() - g = 0"""
        self.F_right = self.F_oneside(*self.get_slices_right())
        self.F_left = self.F_oneside(*self.get_slices_left())
        if self.index == 0:
            self.F = self.F_right
        elif self.index == N - 1:
            self.F = self.F_left
        else:
            self.F = sp.vstack([self.F_left, self.F_right])
        self.sigmas = np.zeros(self.F.shape[0])

    def F_oneside(self, starts, ends):
        """Picks the right node of the clique."""
        n_constraints = sum(
            (end[0] - start[0]) * (end[1] - start[1])
            for start, end in zip(starts, ends)
        )
        assert n_constraints == self.num_overlap
        counter = 0
        i_list = []
        j_list = []
        data = []  # for testing only
        for start, end in zip(starts, ends):
            for i in range(start[0], end[0]):
                for j in range(start[1], end[1]):
                    i_list.append(counter)
                    j_list.append(i * self.X_dim + j)
                    counter += 1
                    if self.X is not None:
                        data.append(self.X[i, j])
        assert counter == n_constraints
        F_oneside = sp.csr_array(
            ([1] * len(i_list), (i_list, j_list)),
            shape=(n_constraints, self.X_var.size),
        )
        if self.X is not None:
            np.testing.assert_allclose(F_oneside @ self.X.flatten(), data)
        return F_oneside

    def generate_overlap_slices(self, N):
        if self.index > 0:  # the RIGHT side of the left node is overlapping
            self.left_slice_start, self.left_slice_end = self.get_slices_right()
        else:
            self.left_slice_start = self.left_slice_end = [[0, 0]]
        if self.index < N - 1:  # the LEFT side of the right node is overlapping
            self.right_slice_start, self.right_slice_end = self.get_slices_left()
        else:
            self.right_slice_start = self.right_slice_end = [[0, 0]]

    def get_slices_right(self):
        """Picks the right part of a node"""
        left_slice_start = [[0, 1 + self.x_dim]]  # i_start, i_end
        left_slice_end = [[1, 1 + 2 * self.x_dim]]
        if self.constrain_all:
            left_slice_start += [[1 + self.x_dim, 1 + self.x_dim]]
            left_slice_end += [[1 + 2 * self.x_dim, 1 + 2 * self.x_dim]]
        return left_slice_start, left_slice_end

    def get_slices_left(self):
        """Picks the left part of a node"""
        right_slice_start = [[0, 1]]
        right_slice_end = [[1, 1 + self.x_dim]]
        if self.constrain_all:
            right_slice_start += [[1, 1]]
            right_slice_end += [[1 + self.x_dim, 1 + self.x_dim]]
        return right_slice_start, right_slice_end

    def current_error(self):
        return np.sum(np.abs(self.F @ self.X_new - self.g))

    def get_constraints_cvxpy(self, X):
        return [cp.trace(A_k @ X) == b_k for A_k, b_k in zip(self.A_list, self.b_list)]

    def get_objective_cvxpy(self, X, rho_k):
        if np.ndim(rho_k) > 0:
            return cp.Minimize(
                cp.trace(self.Q @ X)
                + self.sigmas.T @ (self.F @ X.flatten() - self.g)
                + 0.5
                * cp.norm2(cp.multiply(self.F @ X.flatten() - self.g, np.sqrt(rho_k)))
                ** 2
            )
        else:
            return cp.Minimize(
                cp.trace(self.Q @ X)
                + self.sigmas.T @ (self.F @ X.flatten() - self.g)
                + 0.5 * rho_k * cp.norm2(self.F @ X.flatten() - self.g) ** 2
            )
