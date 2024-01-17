import itertools
from copy import deepcopy

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cert_tools.base_clique import BaseClique

CONSTRAIN_ALL_OVERLAP = False


class CliqueADMM(BaseClique):
    def __init__(
        self,
        Q,
        A_list: list,
        b_list: list,
        var_dict: dict,
        index: int = 0,
        N: int = 0,
        hom="l",
        x_dim: int = 0,
    ):
        super().__init__(
            Q=Q,
            A_list=A_list,
            b_list=b_list,
            var_dict=var_dict,
            hom=hom,
        )
        self.x_dim = x_dim
        self.constrain_all = CONSTRAIN_ALL_OVERLAP
        self.status = 0

        self.X_var = cp.Variable((self.X_dim, self.X_dim), PSD=True)
        self.X_new = None
        self.Z_new = None
        self.Z_prev = None

        self.F = None
        self.g = None

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
        # left and right are Z matrix.
        if ((left is not None) and (left.shape[0] == 1 + self.x_dim)) or (
            (right is not None) and (right.shape[0] == 1 + self.x_dim)
        ):
            self.generate_g_from_shift(left, right, shift=0)
        # left and right are X matrix.
        else:
            self.generate_g_from_shift(left, right, shift=self.x_dim)

    def generate_F(self, left=None, right=None):
        """Generate matrix for overlap constraints: F @ X.flatten() - g = 0"""
        Fs = []
        if self.constrain_all:
            Nx = self.x_dim * (self.x_dim + 1) // 2
            if left is not None:
                F = np.zeros((Nx + self.x_dim, self.X.shape[0] ** 2))
                for k, (i, j) in enumerate(
                    itertools.combinations_with_replacement(range(self.x_dim), 2)
                ):
                    # the upper-left corner of current estimate ...
                    here = np.zeros(self.X.shape)
                    here[1 + i, 1 + j] = 1.0
                    F[k, :] = here.flatten()
                for i in range(self.x_dim):
                    F[Nx + i] = here.flatten()
                Fs.append(F)
            if right is not None:
                F = np.zeros((Nx + self.x_dim, self.X.shape[0] ** 2))
                for k, (i, j) in enumerate(
                    itertools.combinations_with_replacement(range(self.x_dim), 2)
                ):
                    # the lower corner of current estimate ...
                    here = np.zeros(self.X.shape)
                    here[1 + self.x_dim + i, 1 + self.x_dim + j] = 1.0
                    F[k, :] = here.flatten()
                    # should equal the lower corner of left estimate
                for i in range(self.x_dim):
                    here = np.zeros(self.X.shape)
                    here[0, 1 + self.x_dim + i] = 1.0
                    F[Nx + i] = here.flatten()
                Fs.append(F)
        else:
            if left is not None:
                F = np.zeros((self.x_dim, self.X_var.shape[0]))
                # pick left part of the current clique
                F[range(self.x_dim), range(1, 1 + self.x_dim)] = 1.0
                Fs.append(F)
            if right is not None:
                F = np.zeros((self.x_dim, self.X_var.shape[0]))
                # pick right part of the current clique
                F[range(self.x_dim), range(1 + self.x_dim, 1 + 2 * self.x_dim)] = 1.0
                Fs.append(F)
        self.F = np.vstack(Fs)
        self.sigmas = np.zeros(self.F.shape[0])

    def generate_g_from_shift(self, left=None, right=None, shift=0):
        # left and right can either be X or Z variables.
        # If they are Z, then shift is zero and things are easy.
        # If they are X, then we need a shift for the left variable
        gs = []
        if self.constrain_all:
            if left is not None:
                for k, (i, j) in enumerate(
                    itertools.combinations_with_replacement(range(self.x_dim), 2)
                ):
                    # should equal the lower-right corner of left estimate
                    gs.append(left[1 + shift + i, 1 + shift + j])

                for i in range(self.x_dim):
                    gs.append(left[0, 1 + shift + i])
            if right is not None:
                for k, (i, j) in enumerate(
                    itertools.combinations_with_replacement(range(self.x_dim), 2)
                ):
                    gs.append(right[1 + i, 1 + j])
                for i in range(self.x_dim):
                    gs.append(right[0, 1 + i])
        else:
            if left is not None:
                gs += list(left[1 + shift : 1 + shift + self.x_dim, 0])
            if right is not None:
                gs += list(right[1 : 1 + self.x_dim, 0])
        self.g = cp.hstack(gs)

    def generate_overlap_slices(self):
        self.left_slice_start = [[0, 1 + self.x_dim]]
        self.left_slice_end = [[1, 1 + 2 * self.x_dim]]
        self.right_slice_start = [[0, 1]]
        self.right_slice_end = [[1, 1 + self.x_dim]]
        if self.constrain_all:
            self.left_slice_start += [[1 + self.x_dim, 1 + self.x_dim]]
            self.left_slice_end += [[1 + 2 * self.x_dim, 1 + 2 * self.x_dim]]
            self.right_slice_start += [1, 1]
            self.right_slice_end += [[1 + self.x_dim, 1 + self.x_dim]]

    def get_dual_residual(self):
        # we are only interested in the left part of Z here.
        return (self.Z_prev - self.Z_new).flatten()

    def update_Z(self):
        self.Z_prev = deepcopy(self.Z_new)
        self.Z_new = np.hstack(
            [
                np.vstack([1, self.X_new[[0], 1 + self.x_dim :]]),
                np.vstack(
                    [
                        self.X_new[1 + self.x_dim :, [0]],
                        self.X_new[1 + self.x_dim :, 1 + self.x_dim :],
                    ]
                ),
            ]
        )

    def current_error(self):
        return np.sum(np.abs(self.evaluate_F(self.X_new)))

    def evaluate_F(self, X, g=None):
        if g is None:
            g = self.g.value
        if self.F.shape[1] == X.shape[0]:
            return self.F @ X[:, 0] - g
        elif self.F.shape[1] == X.shape[0] ** 2:
            return self.F @ X.flatten() - g
        else:
            raise ValueError(self.F.shape)

    def get_constraints_cvxpy(self, X):
        return [cp.trace(A_k @ X) == b_k for A_k, b_k in zip(self.A_list, self.b_list)]

    def get_objective_cvxpy(self, X, rho_k):
        return cp.Minimize(
            cp.trace(self.Q @ X)
            + self.sigmas.T @ self.evaluate_F(X)
            + 0.5 * rho_k * cp.norm2(self.evaluate_F(X)) ** 2
        )
