from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from pandas import DataFrame, read_pickle
from poly_matrix import PolyMatrix
from pylgmath import so3op

from cert_tools import solve_sdp_mosek
from cert_tools.admm_clique import ADMMClique
from cert_tools.base_clique import BaseClique
from cert_tools.sparse_solvers import solve_oneshot

# Global Defaults
ER_MIN = 1e6


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

    def __init__(self, N=10, sigma=1e-4, seed=0):
        np.random.seed(seed)
        # generate ground truth poses
        aaxis_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(N, 3, 1))
        R_gt = so3op.vec2rot(aaxis_ab_rand)
        # Associated variable list
        self.var_list = {"h": 1}
        for i in range(N):
            self.var_list[i] = 9
        # Generate Measurements as a dictionary on tuples
        # First pose is locked, loop starts at second variable
        self.R_meas = {}
        for i in range(0, N):
            R_pert = so3op.vec2rot(sigma * np.random.randn(3, 1))
            if i == N - 1:
                j = 1
            else:
                j = i + 1
            self.R_meas[(i, j)] = R_pert @ R_gt[i] @ R_gt[j].T
        # Store data
        self.R_gt = R_gt
        self.N = N
        self.sigma = sigma
        # Locked Pose
        self.locked_pose = 0
        # Generate cost matrix
        self.cost = self.get_cost_matrix()
        # Generate Constraints
        self.constraints = self.get_constraint_matrices()

    def get_cost_matrix(self) -> PolyMatrix:
        """Get the cost matrix associated with the problem. Assume equal weighting
        for all measurments
        """
        Q = PolyMatrix()
        # Construct matrix from measurements
        for i, j in self.R_meas.keys():
            Q += self.get_rel_cost_mat(i, j)

        # # Add prior measurement on first pose (tightens the relaxation)
        # Q += self.get_prior_cost_mat(self, 1)

        return Q

    def get_rel_cost_mat(self, i, j) -> PolyMatrix:
        """Get cost representation for relative rotation measurement"""
        Q = PolyMatrix()
        Q[i, j] += -np.kron(np.eye(3), self.R_meas[(i, j)])
        Q[i, i] += 2 * sp.eye(9)
        Q[j, j] += 2 * sp.eye(9)
        return Q

    def get_prior_cost_mat(self, index, weight=1) -> PolyMatrix:
        """Get cost representation for prior measurement"""
        weight = self.N ^ 2
        index = 1
        Q = PolyMatrix()
        Q["h", "h"] += 6 * weight
        Q["h", index] += self.R_gt[index].reshape((9, 1), order="F").T * weight
        return Q

    def get_constraint_matrices(self):
        """Generate all constraints for the problem"""
        constraints = []
        for i in range(self.N):
            constraints += self.get_O3_constraints(i)
            # constraints += self.get_handedness_constraints(i)
            # constraints += self.get_row_col_constraints(i)
            if i == self.locked_pose:
                constraints += self.get_locking_constraint(i)
        # Homogenizing Constraint
        constraints += self.get_homog_constraint()

        return constraints

    def get_locking_constraint(self, index):
        """Get constraint that locks a particular pose to its ground truth value
        rather than adding a prior cost term. This should remove the gauge freedom
        from the problem, giving a rank-1 solution"""
        r_gt = self.R_gt[index].reshape((9, 1), order="F")
        constraints = []
        for k in range(9):
            A = PolyMatrix()
            e_k = np.zeros((1, 9))
            e_k[0, k] = 1
            A["h", index] = e_k / 2
            A["h", "h"] = -r_gt[k]
            constraints += [(A, 0.0)]
        return constraints

    @staticmethod
    def get_O3_constraints(index):
        """Generate O3 constraints for the problem"""
        constraints = []
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
                A[index, index] = np.kron(np.eye(3), E)
                A["h", "h"] = -b
                constraints += [(A, 0.0)]
        return constraints

    @staticmethod
    def get_handedness_constraints(index):
        """Generate Handedness Constraints - Equivalent to the determinant =1
        constraint for rotation matrices. See Tron,R et al:
        On the Inclusion of Determinant Constraints in Lagrangian Duality for 3D SLAM"""
        constraints = []
        i, j, k = 0, 1, 2
        for col_ind in range(3):
            l, m, n = 0, 1, 2
            for row_ind in range(3):
                # Define handedness matrix and vector
                mat = np.zeros((9, 9))
                mat[3 * j + m, 3 * k + n] = 1 / 2
                mat[3 * j + n, 3 * k + m] = -1 / 2
                mat = mat + mat.T
                vec = np.zeros((9, 1))
                vec[i * 3 + l] = -1 / 2
                # Create constraint
                A = PolyMatrix()
                A[index, index] = mat
                A[index, "h"] = vec
                constraints += []
                # cycle row indices
                l, m, n = m, n, l
            # Cycle column indicies
            i, j, k = j, k, i
        return constraints

    @staticmethod
    def get_row_col_constraints(index):
        """Generate constraint that every row vector length equal every column vector length"""
        constraints = []
        for i in range(3):
            for j in range(3):
                A = PolyMatrix()
                c_col = np.zeros(9)
                ind = 3 * j + np.array([0, 1, 2])
                c_col[ind] = np.ones(3)
                c_row = np.zeros(9)
                ind = np.array([0, 3, 6]) + i
                c_row[ind] = np.ones(3)
                A[index, index] = np.diag(c_col - c_row)
                constraints += [(A, 0.0)]
        return constraints

    @staticmethod
    def get_homog_constraint():
        """generate homogenizing constraint"""
        A = PolyMatrix()
        A["h", "h"] = 1
        return [(A, 1.0)]

    def split_graph(self, edge=(4, 5)):
        """Introduce a new variable to make the problem chordal"""
        # Replace loop measurement with meas to new var.
        # New variables identified by the fact that they are strings with "s"
        split_var = str(edge[1]) + "s"
        self.R_meas[(edge[0], split_var)] = self.R_meas.pop(edge)
        # Add new variable
        self.var_list[split_var] = 9
        # Store edge that is associated with ADMM
        self.admm_edge = (split_var, edge[1])
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
        return self.convert_sdp_to_rot(X)

    def chordal_admm(self, tol_res=1e-4, decompose=False):
        """Uses ADMM to convert the SDP into a CHORDAL problem. A new variable is added
        to break the loop topology and consensus constraints are used to force it to
        be consistent with first variable.

        Args:
            tol_res (_type_, optional): tolerance for consensus constraint residual.
            Defaults to 1e-5.

        Returns:
            R : List of rotations
        """
        # SPLIT NON-CHORDAL GRAPH
        edge = (4, 5)
        self.split_graph(edge=edge)
        if decompose:
            # Generate clique matrices
            cliques = self.get_cliques(check_valid=True)
            # find cliques that contain the ADMM variables
            # TODO Need to find a more elegant way to do this
            var1, var2 = self.admm_edge
            admm_cliques = [None] * 2
            for clique in cliques:
                if var1 in clique.var_dict.keys():
                    admm_cliques[0] = clique
                if var1 in clique.var_dict.keys():
                    admm_cliques[1] = clique
            admm_cliques_stored = deepcopy(admm_cliques)
        else:
            Cost = self.cost.get_matrix(self.var_list)
            Constraints = [
                (A.get_matrix(self.var_list), b) for A, b in self.constraints
            ]
        # INIITIALIZE
        U = {
            self.admm_edge[0]: np.zeros((3, 3)),
            self.admm_edge[1]: np.zeros((3, 3)),
        }  # Lagrange Multipliers
        Z = np.zeros((3, 3))  # Concensus Variable
        rho = 0.1
        res_norm = np.inf
        max_iter = 100
        n_iter = 1
        R = None
        # ADMM Loop
        print("Starting ADMM Loop:")
        while res_norm > tol_res and n_iter < max_iter:
            # Compute the augmented Lagrangian cost terms
            F = self.get_aug_lagr_factors(U, Z, rho)
            if decompose:
                # Update costs for cliques that contain split vars
                admm_cliques[0].Q = admm_cliques_stored[0].Q + F[0].get_matrix(
                    admm_cliques[0].var_dict
                )
                admm_cliques[1].Q = admm_cliques_stored[1].Q + F[1].get_matrix(
                    admm_cliques[1].var_dict
                )
                # Solve Decomposed SDP
                X_list_k, info = solve_oneshot(cliques, use_fusion=True, verbose=False)
                # Retreive Solution
                R = self.convert_cliques_to_rot(
                    X_list=X_list_k, cliques=cliques, er_min=1e4
                )
            else:
                # Solve the SDP and convert to rotation matrices
                F_mat = (F[0] + F[1]).get_matrix(self.var_list)
                X, info = solve_sdp_mosek(
                    Q=Cost + F_mat, Constraints=Constraints, adjust=False, verbose=False
                )
                R = self.convert_sdp_to_rot(X)
            # Update Consensus Variable
            ind1, ind2 = self.admm_edge
            Z = (R[ind1] + R[ind2] + U[ind1] + U[ind2]) / 2
            # Update Lagrange Multipliers
            res = [R[ind1] - Z, R[ind2] - Z]
            U[ind1] += res[0]
            U[ind2] += res[1]

            # Update stopping criterion
            res_norm = np.linalg.norm(res[0], "fro") + np.linalg.norm(res[1], "fro")
            # Print stuff
            print(f"{n_iter}:\t{res_norm}")
            n_iter += 1

        # Return solution as list
        R_list = [R[i] for i in range(self.N)]
        return R_list

    def get_aug_lagr_factors(self, U, Z, rho):
        """Retrieve augmented lagrangian penalty terms"""
        i, j = self.admm_edge
        F1 = PolyMatrix()
        s1 = (U[i] - Z).reshape((9, 1), order="F")
        F1["h", i] = s1.T
        F1["h", "h"] += 3 + s1.T @ s1
        F1 *= rho / 2
        F2 = PolyMatrix()
        s2 = (U[j] - Z).reshape((9, 1), order="F")
        F2["h", j] = s2.T
        F2["h", "h"] += 3 + s2.T @ s2
        F2 *= rho / 2

        return [F1, F2]

    def get_cliques(self, check_valid=False):
        """Generate the list of cliques associated with the rotation synch problem
        In general, there will be a clique for each edge in the measurement
        graph.
        NOTE: Eventually this will depend on the aggregate sparsity graph."""
        cliques = []

        cost_sum = PolyMatrix()
        for edge in self.R_meas.keys():
            i, j = edge
            # Variables
            var_dict = {i: 9, j: 9, "h": 1}
            # Constraints
            constraints = self.get_O3_constraints(i)
            constraints += self.get_O3_constraints(j)
            constraints += self.get_homog_constraint()
            if i == self.locked_pose:
                constraints += self.get_locking_constraint(i)
            elif j == self.locked_pose:
                constraints += self.get_locking_constraint(j)

            A_list, b_list = zip(*constraints)
            A_list = [A.get_matrix(var_dict) for A in A_list]
            # Cost Functions
            cost = self.get_rel_cost_mat(i, j)
            # Debug - sum the cost
            cost_sum += cost
            # Get matrix version of cost
            cost = cost.get_matrix(var_dict)

            # Build clique
            cliques += [
                ADMMClique(
                    cost,
                    A_list=A_list,
                    b_list=b_list,
                    var_dict=var_dict,
                    hom="h",
                    index=i,
                    N=2,
                )
            ]

        if check_valid:
            Q1 = cost_sum.get_matrix(self.var_list)
            Q2 = self.cost.get_matrix(self.var_list)
            np.testing.assert_allclose(Q1.todense(), Q2.todense())

        return cliques

    def convert_sdp_to_rot(self, X, er_min=ER_MIN):
        """
        Converts a solution matrix to a list of rotations.

        Parameters:
        - X: numpy.ndarray
            The solution matrix.

        Returns:
        - R: list
            A list of rotation matrices.
        """
        # Extract via SVD
        U, S, V = np.linalg.svd(X)
        # Eigenvalue ratio check
        assert S[0] / S[1] > er_min, ValueError("SDP is not Rank-1")
        x = U[:, 0] * np.sqrt(S[0])
        # Convert to list of rotations
        # R_vec = x[1:]
        R_vec = X[1:, 0]
        R_block = R_vec.reshape((3, -1), order="F")
        # Check determinant - Since we just take the first column, its possible for
        # the entire solution to be flipped
        if np.linalg.det(R_block[:, :3]) < 0:
            sign = -1
        else:
            sign = 1

        R = {}
        cnt = 0
        for key in self.var_list.keys():
            if "h" == key:
                continue
            R[key] = sign * R_block[:, 3 * cnt : 3 * (cnt + 1)]
            cnt += 1
        return R

    def convert_cliques_to_rot(self, X_list, cliques, er_min=ER_MIN):
        """Function for converting clique SDP Solutions back to a solution to
        the original problem"""
        R = {}
        for iClq, X in enumerate(X_list):
            # Check Tightness
            if er_min > 0:
                evals = np.linalg.eigvalsh(X)
                er = evals[-1] / evals[-2]
                assert er > er_min, ValueError("Clique is not Rank-1")
            # Determine index of homogenizing variable
            ind = 0
            for key, val in cliques[iClq].var_dict.items():
                if not key == cliques[iClq].hom:
                    ind += val
                else:
                    break

            # Get homogenious column from SDP matrix
            R_vec = X[:, ind]
            R_vec = np.delete(R_vec, ind)
            R_block = R_vec.reshape((3, -1), order="F")
            # Check determinant - Since we just take the first column, its possible for
            # the entire solution to be flipped
            if np.linalg.det(R_block[:, :3]) < 0:
                R_block = -R_block
            # Use clique variables to assign rotations.
            cnt = 0
            for key in self.var_list.keys():
                if "h" == key:
                    continue
                R[key] = R_block[:, 3 * cnt : 3 * (cnt + 1)]
                cnt += 1
        return R

    def check_solution(self, R, get_rmse=False, atol=5e-3, rtol=0.0):

        rmse = 0.0
        for i in range(self.N):
            if not get_rmse:
                np.testing.assert_allclose(
                    R[i],
                    self.R_gt[i],
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"Solution not close to ground truth for Rotation {i}",
                )
            else:
                diff = so3op.rot2vec(np.linalg.inv(R[i]) @ self.R_gt[i])
                rmse += np.linalg.norm(diff) ** 2
        if get_rmse:
            rmse = np.sqrt(rmse / len(R))
            return rmse

    def plot_matrices(self):
        Cost = self.cost.get_matrix(self.var_list)
        plt.matshow(Cost.todense())
        A_all = np.sum([A.get_matrix(self.var_list) for A, b in self.constraints])
        plt.matshow(A_all.todense())
        plt.show()


def test_chord_admm(decompose=False, N=10):
    prob = RotSynchLoopProblem(N=N)
    # Solve SDP
    R = prob.chordal_admm(decompose=decompose)
    # Check solution
    prob.N -= 1
    prob.check_solution(R)


def test_nonchord_sdp(N=10):
    prob = RotSynchLoopProblem(N=N)
    # Solve SDP
    R = prob.solve_sdp()
    # Check solution
    prob.check_solution(R)


def compare_solvers():
    prob_sizes = [10, 20, 50, 100, 200, 500]
    # prob_sizes = [10, 20]
    data = []
    for N in prob_sizes:
        prob = RotSynchLoopProblem(N=N)
        # Solve via SDP
        time_start = time()
        R_sdp = prob.solve_sdp()
        time_end = time()
        rmse_sdp = prob.check_solution(R_sdp, get_rmse=True)
        time_sdp = time_end - time_start
        # Solve via ADMM
        time_start = time()
        R_admm = prob.chordal_admm(decompose=True)
        time_end = time()
        prob.N = prob.N - 1  # Hack
        rmse_admm = prob.check_solution(R_admm, get_rmse=True)
        time_admm = time_end - time_start
        # store values
        data += [
            {
                "prob_size": N,
                "time_sdp": time_sdp,
                "rmse_sdp": rmse_sdp,
                "time_admm": time_admm,
                "rmse_admm": rmse_admm,
            }
        ]
    df = DataFrame(data)
    df.to_pickle("stored_result.pkl")


def compare_solvers_plot():
    df = read_pickle("stored_result.pkl")
    plt.figure()
    plt.loglog(df.prob_size, df.time_sdp, ".-", label="SDP ")
    plt.loglog(df.prob_size, df.time_admm, ".-", label="ADMM dSDP ")
    plt.ylabel("Run Time [s]")
    plt.xlabel("Number of Poses")
    plt.title("Runtime Comparison")
    plt.legend()

    plt.figure()
    plt.loglog(df.prob_size, df.rmse_sdp, ".-", label="SDP ")
    plt.loglog(df.prob_size, df.rmse_admm, ".-", label="ADMM dSDP ")
    plt.ylabel("RMSE [rad]")
    plt.xlabel("Number of Poses")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # test_nonchord_sdp(N=10)
    # prob = RotSynchLoopProblem()
    # prob.plot_matrices()
    # prob.split_graph()
    # prob.plot_matrices()
    # test_chord_sdp()

    # ADMM without decomposition
    # test_chord_admm(decompose=False, N=10)

    # Test clique generation
    # prob = RotSynchLoopProblem()
    # cliques = prob.get_cliques()

    # ADMM with decomposition
    test_chord_admm(decompose=True, N=10)

    # Compare solvers
    # compare_solvers()
    # compare_solvers_plot()

    # print("done")
