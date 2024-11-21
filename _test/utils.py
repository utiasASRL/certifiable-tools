import numpy as np
from poly_matrix import PolyMatrix
from pylgmath import so3op

from cert_tools import HomQCQP

# Global Defaults
ER_MIN = 1e6


class RotSynchLoopProblem(HomQCQP):
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
        meas_dict (dict): Dictionary of noisy measurements
        cost (PolyMatrix): Cost matrix for the SDP
        constraints (list): List of O(3) constraints for the SDP
    """

    """Rotation synchronization problem configured in a loop (non-chordal)    
        """

    def __init__(self, N=10, sigma=1e-3, loop_pose=3, locked_pose=0, seed=0):
        super().__init__()

        np.random.seed(seed)
        # generate ground truth poses
        aaxis_ab_rand = np.random.uniform(-np.pi / 2, np.pi / 2, size=(N, 3, 1))
        R_gt = so3op.vec2rot(aaxis_ab_rand)
        # Associated variable list
        self.var_sizes = {"h": 1}
        for i in range(N):
            self.var_sizes[str(i)] = 9
        # Generate Measurements as a dictionary on tuples
        self.loop_pose = str(loop_pose)  # Loop relinks to chain at this pose
        self.locked_pose = str(locked_pose)  # Pose locked at this pose
        self.meas_dict = {}
        for i in range(0, N):
            R_pert = so3op.vec2rot(sigma * np.random.randn(3, 1))
            if i == N - 1:
                if loop_pose > 0:
                    j = loop_pose
                else:
                    continue
            else:
                j = i + 1
            self.meas_dict[(str(i), str(j))] = R_pert @ R_gt[i] @ R_gt[j].T
        # Store data
        self.R_gt = R_gt
        self.N = N
        self.sigma = sigma
        # Define obj and constraints
        self.C = self.define_objective()
        self.As = self.define_constraints()

    def define_objective(self) -> PolyMatrix:
        """Get the cost matrix associated with the problem. Assume equal weighting
        for all measurments
        """
        Q = PolyMatrix()
        # Construct matrix from measurements
        for i, j in self.meas_dict.keys():
            Q += self.get_rel_cost_mat(i, j)

        # # Add prior measurement on first pose (tightens the relaxation)
        # Q += self.get_prior_cost_mat(self, 1)

        return Q

    def get_rel_cost_mat(self, i, j) -> PolyMatrix:
        """Get cost representation for relative rotation measurement"""
        Q = PolyMatrix()
        if (i, j) in self.meas_dict.keys():
            meas = self.meas_dict[(i, j)]
        else:
            meas = self.meas_dict[(j, i)]
        Q[i, j] = -np.kron(np.eye(3), meas)
        Q[i, i] = 2 * np.eye(9)
        Q[j, j] = 2 * np.eye(9)
        return Q

    def get_prior_cost_mat(self, index, weight=1) -> PolyMatrix:
        """Get cost representation for prior measurement"""
        weight = self.N ^ 2
        index = 1
        Q = PolyMatrix()
        Q["h", "h"] += 6 * weight
        Q["h", index] += self.R_gt[index].reshape((9, 1), order="F").T * weight
        return Q

    def define_constraints(self) -> list[PolyMatrix]:
        """Generate all constraints for the problem"""
        constraints = []
        for key in self.var_sizes.keys():
            if key == "h":
                continue
            else:
                # Otherwise add rotation constriants
                constraints += self.get_O3_constraints(key)
                constraints += self.get_handedness_constraints(key)
                constraints += self.get_row_col_constraints(key)
                # lock the appropriate pose
                if key == self.locked_pose:
                    constraints += self.get_locking_constraint(key)

        return constraints

    def get_locking_constraint(self, index):
        """Get constraint that locks a particular pose to its ground truth value
        rather than adding a prior cost term. This should remove the gauge freedom
        from the problem, giving a rank-1 solution"""

        r_gt = self.R_gt[int(index)].reshape((9, 1), order="F")
        constraints = []
        for k in range(9):
            A = PolyMatrix()
            e_k = np.zeros((1, 9))
            e_k[0, k] = 1
            A["h", index] = e_k / 2
            A["h", "h"] = -r_gt[k]
            constraints.append(A)
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
                constraints.append(A)
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
                constraints.append(A)
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
                constraints.append(A)
        return constraints

    @staticmethod
    def get_homog_constraint():
        """generate homogenizing constraint"""
        A = PolyMatrix()
        A["h", "h"] = 1
        return [(A, 1.0)]

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
        for key in self.var_sizes.keys():
            if "h" == key:
                continue
            R[key] = sign * R_block[:, 3 * cnt : 3 * (cnt + 1)]
            cnt += 1
        return R


def get_chain_rot_prob(N=10, locked_pose=0):
    return RotSynchLoopProblem(N=N, loop_pose=-1, locked_pose=locked_pose)


def get_loop_rot_prob():
    return RotSynchLoopProblem()
