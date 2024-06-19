import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix
from pylgmath import Rotation, so3op

from cert_tools import solve_sdp_mosek


class RotSynchLoopProblem:
    """Class to generate and solve a rotation synchronization problem with loop
    constraints. The problem is generated with ground truth rotations and noisy
    measurements between them. The goal is to recover the ground truth rotations
    using a semidefinite programming approach. The loop constraints are encoded
    as O(3) constraints in the SDP.
    Problem is vectorized so that we can add a homogenization variable.

    Attributes:
        N (int): Number of poses in the problem
        sigma (float): Standard deviation of noise in measurements
        R_gt (np.array): Ground truth rotations
        R_meas (dict): Dictionary of noisy measurements
        cost (PolyMatrix): Cost matrix for the SDP
        constraints (list): List of O(3) constraints for the SDP
    """

    """Rotation synchronization problem configured in a loop (non-chordal)    
        """

    def __init__(self, N=100, sigma=1e-3, seed=0):
        np.random.seed(seed)
        # generate ground truth poses
        aaxis_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(N, 3, 1))
        R_gt = so3op.vec2rot(aaxis_ab_rand)
        # Associated variable list
        self.var_list = {"h": 1}
        for i in range(N):
            self.var_list[i] = 9
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
            Q[i, j] += -np.kron(np.eye(3), self.R_meas[(i, j)])
            Q[i, i] += 2 * sp.eye(9)
            Q[j, j] += 2 * sp.eye(9)

        # Add prior measurement on first pose (tightens the relaxation)
        weight = 1
        index = 1
        Q["h", "h"] += 6 * weight
        Q["h", index] += self.R_gt[index].reshape((9, 1), order="F").T * weight

        return Q

    def get_constraint_matrices(self):
        """Generate O3 constraints for the problem"""
        constraints = []
        for i in range(self.N):
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
                    A[i, i] = np.kron(np.eye(3), E)
                    A["h", "h"] = -b
                    constraints += [(A, 0.0)]

        # Homogenizing Constraint
        A = PolyMatrix()
        A["h", "h"] = 1
        constraints += [(A, 1.0)]

        return constraints

    def convert_to_chordal(self):
        """Introduce a new variable to make the problem chordal"""
        # Replace loop measurement with meas to new var
        self.R_meas[(self.N - 1, self.N)] = self.R_meas.pop((self.N - 1, 0))
        # Add new variable
        self.var_list[self.N] = 9
        self.N += 1
        # Regenerate Cost and Constraints
        self.cost = self.get_cost_matrix()
        self.constraints = self.get_constraint_matrices()

    def solve_sdp(self):
        """Solve non-chordal SDP for PGO problem without using ADMM"""
        # Convert to sparse matrix from polymatrix
        Cost = self.cost.get_matrix(self.var_list)
        Constraints = [(A.get_matrix(self.var_list), b) for A, b in self.constraints]
        # Solve non-Homogenized SDP
        X, info = solve_sdp_mosek(
            Q=Cost, Constraints=Constraints, adjust=False, verbose=True
        )
        # Extract solution
        return self.convert_soln_to_rot(X)

    def convert_soln_to_rot(self, X):
        """
        Converts a solution matrix to a list of rotations.

        Parameters:
        - X: numpy.ndarray
            The solution matrix.

        Returns:
        - R: list
            A list of rotation matrices.
        """
        # Rank Check
        assert np.linalg.matrix_rank(X, tol=1e-6) == 1, ValueError("SDP is not Rank-1")
        # Extract via SVD
        U, S, V = np.linalg.svd(X)
        x = U[:, 0] * np.sqrt(S[0])
        # Convert to list of rotations
        R_vec = x[1:]
        R_block = R_vec.reshape((3, -1), order="F")
        R = [R_block[:, 3 * i : 3 * (i + 1)] for i in range(self.N)]
        return R

    def chordal_admm(self, tol_res=1e-5):
        """Uses ADMM to convert the SDP into a CHORDAL problem. A new variable is added
        to break the loop topology and consensus constraints are used to force it to
        be consistent with first variable.

        Args:
            tol_res (_type_, optional): tolerance for consensus constraint residual.
            Defaults to 1e-5.

        Returns:
            R : List of rotations
        """
        # Retrieve the chordal version of the problem with the additional variable.
        self.convert_to_chordal()
        Cost = self.cost.get_matrix(self.var_list)
        Constraints = [(A.get_matrix(self.var_list), b) for A, b in self.constraints]
        # Initialize variables
        U = np.zeros((2, 3, 3))  # Lagrange Multiplier
        Z = np.zeros((3, 3))  # Concensus Variable
        rho = 0.1
        res_norm = np.inf
        max_iter = 100
        n_iter = 1
        R = None
        # ADMM Loop
        while res_norm > tol_res and n_iter < max_iter:
            # Compute the augmented Lagrangian cost terms
            s1 = (U[0] - Z).reshape((9, 1), order="F")
            s2 = (U[1] - Z).reshape((9, 1), order="F")
            F = PolyMatrix()
            F["h", 0] = s1.T
            F["h", self.N - 1] = s1.T
            F["h", "h"] += 6 + s1.T @ s1 + s2.T @ s2
            # Solve the SDP and convert to rotation matrices
            F_mat = F.get_matrix(self.var_list) * rho / 2
            X, info = solve_sdp_mosek(
                Q=Cost + F_mat, Constraints=Constraints, adjust=False, verbose=False
            )
            R = self.convert_soln_to_rot(X)
            # Update Consensus Variable
            Z = (R[0] + R[-1] + U[0] + U[-1]) / 2
            # Update Lagrange Multiplier
            res = [Z - R[0], Z - R[-1]]
            U += -np.stack(res, 0)
            # Update stopping criterion
            res_norm = np.linalg.norm(res[0], "fro") + np.linalg.norm(res[1], "fro")
            n_iter += 1
            # Print stuff
            print(f"{n_iter}:\t{res_norm}")

        # Return solution
        return R[:-1]

    def check_solution(self, R, atol=5e-3, rtol=0.0):
        for i in range(self.N):
            np.testing.assert_allclose(
                R[i],
                self.R_gt[i],
                atol=atol,
                rtol=rtol,
                err_msg=f"Solution not close to ground truth for Rotation {i}",
            )

    def plot_matrices(self):
        Cost = self.cost.get_matrix(self.var_list)
        plt.matshow(Cost.todense())
        A_all = np.sum([A.get_matrix(self.var_list) for A, b in self.constraints])
        plt.matshow(A_all.todense())
        plt.show()


def test_chord_admm():
    prob = RotSynchLoopProblem()
    # Solve SDP
    R = prob.chordal_admm()
    # Check solution
    prob.N -= 1
    prob.check_solution(R)


def test_nonchord_sdp():
    prob = RotSynchLoopProblem()
    # Solve SDP
    R = prob.solve_sdp()
    # Check solution
    prob.check_solution(R)


if __name__ == "__main__":

    # test_nonchord_sdp()
    # prob = RotSynchLoopProblem()
    # prob.plot_matrices()
    # prob.convert_to_chordal()
    # prob.plot_matrices()
    # test_chord_sdp()
    test_chord_admm()
