import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix
from pylgmath import Rotation, so3op

from cert_tools import solve_sdp_mosek


class RotationLoopProblem:
    def __init__(self, N=6, sigma=1e-3, seed=0):
        """Generate a pose graph example configured in a loop (non-chordal)"""
        np.random.seed(seed)
        # generate ground truth poses
        aaxis_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(N, 3, 1))
        R_gt = so3op.vec2rot(aaxis_ab_rand)
        # Associated variable list
        self.var_list = {}
        for i in range(N):
            self.var_list[i] = 3
        # self.var_list['h'] = 1
        # Generate Measurements as a dictionary on tuples
        self.R_meas = {}
        for i in range(N):
            R_pert = so3op.vec2rot(sigma * np.random.randn(3, 1))
            j = (i + 1) % N
            self.R_meas[(i, j)] = R_pert @ R_gt[i] @ R_gt[j].T
        # Store data
        self.R_gt = R_gt
        self.N = N
        self.sigma = sigma
        # Generate cost matrix
        self.cost = self.get_cost_matrix()
        # Generate Constraints
        self.constraints = self.get_constraint_matrices()

    def get_cost_matrix(self):
        """Get the cost matrix associated with the problem. Assume equal weighting
        for all measurments
        """
        Q = PolyMatrix()
        # Construct matrix from measurements
        for i, j in self.R_meas.keys():
            Q[i, j] = -self.R_meas[(i, j)]
            Q[i, i] += 2 * sp.eye(3)
            Q[j, j] += 2 * sp.eye(3)

        return Q

    def get_constraint_matrices(self):
        """Generate O3 constraints for the problem"""
        constraints = []
        for i in self.var_list.keys():
            for k in range(3):
                for l in range(k, 3):
                    A = PolyMatrix()
                    E = np.zeros((3, 3))
                    if k == l:
                        E[k, l] = 1
                        b = 1.0
                    else:
                        E[k, l] = 1
                        E[l, k] = 1
                        b = 0.0
                    A[i, i] = E
                    constraints += [(A, b)]
        return constraints

    def convert_to_chordal(self):
        """Introduce a new variable to make the problem chordal"""
        # Replace loop measurement with meas to new var
        self.R_meas[(self.N - 1, self.N)] = self.R_meas.pop((self.N - 1, 0))
        # Add new variable
        self.N += 1
        # Regenerate Cost and Constraints
        self.cost = self.get_cost_matrix()
        self.constraints = self.get_constraint_matrices()

    def solve_nonchordal_sdp(self):
        """Solve non-chordal SDP for PGO problem without using ADMM"""
        # Convert to sparse matrix from polymatrix
        Cost = self.cost.get_matrix(self.var_list)
        Constraints = [(A.get_matrix(self.var_list), b) for A, b in self.constraints]
        # Solve non-Homogenized SDP
        X, info = solve_sdp_mosek(
            Q=Cost, Constraints=Constraints, adjust=False, verbose=False
        )
        # Extract solution and align frames
        assert np.linalg.matrix_rank(X, tol=1e-6) == 3, ValueError("SDP is not Rank-3")
        U, S, V = np.linalg.svd(X)
        R = U[:, :3] * np.sqrt(S[:3])
        R_adj = R[:3, :3].T @ self.R_gt[0]
        R = R @ R_adj
        return R

    def plot_matrices(self):
        Cost = self.cost.get_matrix(self.var_list)
        plt.matshow(Cost.todense())
        A_all = np.sum([A.get_matrix(self.var_list) for A, b in self.constraints])
        plt.matshow(A_all.todense())
        plt.show()


def test_nonchord_sdp():
    prob = RotationLoopProblem()
    # Solve SDP
    R = prob.solve_nonchordal_sdp()

    for i in range(prob.N):
        ind = slice(3 * i, 3 * (i + 1))
        diff = so3op.rot2vec(R[ind, :3] @ prob.R_gt[i].T)
        np.testing.assert_allclose(
            diff,
            np.zeros((3, 1)),
            atol=2e-3,
            err_msg="Solution not close to ground truth",
        )


if __name__ == "__main__":
    # test_nonchord_sdp()
    prob = RotationLoopProblem()
    prob.plot_matrices()
    prob.convert_to_chordal()
    prob.plot_matrices()
