import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
import sdprlayers.utils.ess_mat_utils as utils
import torch
from diffcp.cones import unvec_symm, vec_symm
from poly_matrix import PolyMatrix
from sdprlayers import SDPEssMatEst
from sdprlayers.utils.camera_model import CameraModel
from sdprlayers.utils.lie_algebra import se3_exp, so3_exp, so3_log, so3_wedge

from cert_tools.linalg_tools import get_nullspace
from cert_tools.sdp_solvers import (
    solve_low_rank_sdp,
    solve_sdp_fusion,
    solve_sdp_homqcqp,
)


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class EssMatProblem:
    def __init__(self, *args, n_batch=1, n_points=50, tol=1e-12, seed=1, **kwargs):
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        set_seed(seed)
        # Offset distance
        dist = 3
        # Generate problem data
        ts_ts_s, Rs_ts, keys_3d_s = utils.get_gt_setup(
            N_map=n_points,
            N_batch=n_batch,
            traj_type="circle",
            offs=np.array([[0, 0, dist]]).T,
            lm_bound=dist * 0.3,
        )
        # Define Camera
        # self.camera = CameraModel(800, 800, 0.0, 0.0, 0.0)
        self.camera = CameraModel(1, 1, 0.0, 0.0, 0.0)

        # Transforms from source to target
        self.xi = so3_log(torch.tensor(Rs_ts))
        self.xi.requires_grad_(True)
        self.Rs_ts = so3_exp(self.xi)
        ts_ts_s = torch.tensor(ts_ts_s)
        ts_st_t = self.Rs_ts.bmm(-ts_ts_s)
        # Keep track of gradients for translation
        ts_st_t.requires_grad_(True)

        # Keypoints (3D) defined in source frame
        keys_3d_s = torch.tensor(keys_3d_s)[None, :, :].expand(n_batch, -1, -1)
        self.keys_3d_s = keys_3d_s
        # Homogenize coords
        keys_3dh_s = torch.concat(
            [keys_3d_s, torch.ones(n_batch, 1, keys_3d_s.size(2))], dim=1
        )
        # Apply camera to get image points
        src_img_pts = self.camera.camera_model(keys_3dh_s)
        # Get inverse intrinsic camera mat
        K_inv = torch.linalg.inv(self.camera.K)
        K_invs = K_inv.expand(n_batch, 3, 3)
        # Store normalized image coordinates
        self.keypoints_src = K_invs.bmm(src_img_pts)
        # Map to target points
        self.keypoints_trg = self.map_src_to_trg(
            keys_3d_s=keys_3d_s, Rs_ts=self.Rs_ts, ts_st_t=ts_st_t
        )
        # Check the keypoints to ensure good conditioning
        if np.any(np.abs(self.keypoints_src.detach().numpy()) > 1e2):
            raise ValueError("Source keypoints are not well conditioned")
        if np.any(np.abs(self.keypoints_trg.detach().numpy()) > 1e2):
            raise ValueError("Source target are not well conditioned")

        # Generate Scalar Weights
        self.weights = torch.ones(
            self.keypoints_src.size(0), 1, self.keypoints_src.size(2)
        )

        # Normalize the translations
        t_norm = torch.norm(ts_st_t, dim=1, keepdim=True)
        self.ts_st_t_norm = ts_st_t / t_norm
        self.ts_st_t_unnorm = ts_st_t
        # Construct Essential Matrix
        self.Es = self.get_essential(self.ts_st_t_norm, self.xi)

        # Check that the matrix makes sense
        check = 0.0
        for n in range(n_batch):
            check += (
                self.keypoints_trg[0, :, [n]].mT
                @ self.Es[0]
                @ self.keypoints_src[0, :, [n]]
            )

        np.testing.assert_allclose(check.detach(), 0.0, atol=1e-12)

        # Construct solution vectors
        self.sol = torch.cat(
            [
                torch.ones((n_batch, 1, 1)),
                torch.reshape(self.Es, (-1, 9, 1)),  # row-major vectorization
                self.ts_st_t_norm,
            ],
            dim=1,
        )
        K_batch = self.camera.K.expand(1, -1, -1)
        # Initialize layer
        self.layer = SDPEssMatEst(tol=tol, K_source=K_batch, K_target=K_batch)

    def get_essential(self, ts, xi):
        """Return essential matrix associated with a translation and Lie algebra representation of a rotation matrix."""
        return so3_wedge(ts[..., 0]) @ so3_exp(xi)

    def map_src_to_trg(self, keys_3d_s, Rs_ts, ts_st_t):
        """Maps 3D source keypoints to 2D target keypoints."""
        n_batch = self.keys_3d_s.shape[0]
        # Keypoints in target frame
        keys_3d_t = Rs_ts.bmm(keys_3d_s) + ts_st_t
        # homogenize coordinates
        trg_coords = torch.concat(
            [keys_3d_t, torch.ones(n_batch, 1, keys_3d_s.size(2))], dim=1
        )
        # Map through camera model
        trg_img_pts = self.camera.camera_model(trg_coords)
        # Normalize camera coords
        K_inv = torch.linalg.inv(self.camera.K)
        K_invs = K_inv.expand(n_batch, 3, 3)
        keys_2d_t = K_invs.bmm(trg_img_pts)
        return keys_2d_t

    def compute_sdp_solutions(self, sigma_val=10 / 800):
        """Test the objective matrix with no noise"""
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Get low rank solution using chordal sparsity
        Es, Rs, ts, X_lr, rank = self.layer.forward(
            srcs, trgs, self.weights, verbose=True
        )
        X_lr = X_lr[0].detach().numpy()
        x_sol = X_lr[:, [0]]

        # Solve without sparsity
        homQCQP = self.layer.homQCQP
        X_hr, info, solve_time = solve_sdp_homqcqp(homQCQP, tol=1e-12, verbose=True)

        return X_lr, X_hr

    def get_cost_constraints(self):
        # get homogenized qcqp
        homQCQP = self.layer.homQCQP
        # get cost and constraints
        C = homQCQP.C.get_matrix(variables=homQCQP.var_sizes)
        As = homQCQP.As
        constraints = [(A.get_matrix(variables=homQCQP.var_sizes), 0.0) for A in As]
        # Add homogenizing constraint
        h = homQCQP.h
        A_0 = PolyMatrix()
        A_0[h, h] = 1.0
        constraints.append((A_0.get_matrix(variables=homQCQP.var_sizes), 1.0))
        return C, constraints

    def plot_convex_comb(self, X_lr, X_hr):

        # get constraints
        C, constraints = self.get_cost_constraints()

        cost_diff = []
        violation = []
        trace_val = []
        logdet_val = []
        alphas = np.logspace(-5, 0, 20)
        for alpha in alphas:
            X = alpha * X_hr + (1 - alpha) * X_lr
            cost_diff.append(np.abs(np.trace(C @ (X - X_hr))))
            viol = []
            for A, b in constraints:
                viol.append(np.abs(np.trace(A @ X) - b))
            violation.append(np.max(viol))
            trace_val.append(np.trace(X))
            logdet_val.append(-np.log(np.linalg.det(X)))
        # Plot the results
        fig, ax = plt.subplots(3, 1)
        ax[0].semilogx(alphas, trace_val, label="Trace")
        ax[0].set_ylabel("Trace value")
        ax[1].semilogx(alphas, logdet_val, label="Log Det")
        ax[1].set_ylabel("Neg Log-det value")
        plt.legend()
        ax[2].loglog(alphas, cost_diff, label="Change in Cost")
        ax[2].loglog(alphas, violation, label="Violation")
        plt.legend()
        ax[2].set_xlabel("Convex Combination")
        ax[2].set_ylabel("Solution Invariants")
        plt.show()

    def bm_rank_reduction(self, X_0):
        """Rank reduction using Burer-Monteiro method.

        Args:
            X_0 (_type_): Initial solution

        Returns:
            _type_: _description_
        """
        C, Constraints = self.get_cost_constraints()
        # Compute low rank factor
        U, S, V = np.linalg.svd(X_0)
        rank = 4  # np.sum(S > 1e-7)
        Y_0 = U[:, :rank] @ np.sqrt(np.diag(S[:rank]))
        # Append constant cost constraint
        Constraints.append((C, np.trace(C @ X_0)))
        # Solve Rank Reduction with standard SDP
        X_ip, info = solve_sdp_fusion(
            np.eye(X_0.shape[0]),
            Constraints,
            verbose=True,
            tol=1e-8,
            primal=False,
            adjust=False,
        )

        # Solve rank reduction using burer monteiro method.
        Y_opt, info = solve_low_rank_sdp(
            Q=np.eye(C.shape[0]),
            Constraints=Constraints,
            verbose=True,
            rank=rank,
            Y_0=Y_0,
        )

        return Y_opt


def check_feasibility(X, p_opt, C, constraints, tol=1e-3):
    """Check feasibility of the solution."""
    # Check the constraints
    viol = []
    for A, b in constraints:
        viol.append(np.abs(np.trace(A @ X) - b))
    # Check the cost
    viol.append(np.abs(np.trace(C @ X) - p_opt))

    return viol


def rank_reduction(Constraints, X_hr, rank_tol=1e-6, eig_tol=1e-9, max_iter=None):
    """Algorithm that searches for a low rank solution to the SDP problem, given an existing high rank solution.
    Based on the algorithm proposed in "Low-Rank Semidefinite Programming:Theory and Applications by Lemon et al.
    """
    # LOW RANK FACTORIZATION
    # get eigenspace
    vals, vecs = np.linalg.eigh(X_hr)
    # remove zero eigenspace
    r = np.sum(vals > rank_tol)
    n = X_hr.shape[0]
    V = vecs[:, (n - r) :] * np.sqrt(vals[(n - r) :])
    # GET NULLSPACE MATRIX
    Av = compute_Av(Constraints, V)

    # REDUCE RANK
    dim_null = 1
    n_iter = 0
    while dim_null > 0 and (max_iter is None or n_iter < max_iter):
        # Compute null space
        # NOTE: This could be made faster by just computing a single right singular vector in the null space. No need to compute the entire space.
        basis, info = get_nullspace(Av, method="svd", tolerance=rank_tol)
        dim_null = basis.shape[0]
        if dim_null == 0:
            break
        # Get nullspace vector corresponding to the lowest gain eigenvalue
        Delta = unvec_symm(basis[-1], dim=V.shape[1])
        # Compute Eigenspace of Delta
        lambdas, Q = np.linalg.eigh(Delta)
        max_lambda = lambdas[-1]
        # Compute reduced lambdas
        lambdas_red = 1 - lambdas / max_lambda
        # Check which eigenvalues are still nonzero
        inds = lambdas_red > eig_tol
        # Get update matrix
        Q_tilde = Q[:, inds] * np.sqrt(lambdas_red[inds])
        # Update Nullspace matrix
        Av = update_Av(Av, Q_tilde, dim=r)
        # Update Factor
        V = V @ Q_tilde
        r = V.shape[1]
        n_iter += 1

    return V


def compute_Av(Constraints, V):
    """Function to compute the matrix whose nullspace characterizes the optimal solution."""
    Av = []
    for A, b in Constraints:
        Av.append(vec_symm(V.T @ A @ V))
    Av = np.stack(Av)
    return Av


def update_Av(Av, Q_tilde, dim):
    """Update the nullspace matrix. Updating this way is cheaper than reproducing the matrix, because it is performed in the lower dimension."""
    Av_updated = []
    for i, row in enumerate(Av):
        A = unvec_symm(row, dim=dim)
        Av_updated.append(vec_symm(Q_tilde.T @ A @ Q_tilde))
    Av_updated = np.stack(Av_updated)
    return Av_updated


def test_rank_reduction():
    # Create problem
    prob = EssMatProblem(n_batch=1, n_points=50, tol=1e-12)
    # Compute SDP solutions
    X_lr, X_hr = prob.compute_sdp_solutions()
    # Get cost and constraints
    C, Constraints = prob.get_cost_constraints()
    # Optimal Solution
    p_opt = np.trace(C @ X_hr)
    # Get low rank solution
    V = rank_reduction(Constraints, X_hr, rank_tol=1e-2)
    if V[0, 0] < 0:
        V = -V
    X_lr_new = V @ V.T
    # Check Feasibility of low rank solution
    viol = check_feasibility(X_lr_new, p_opt, C, Constraints)
    print("Constraint Violation:")
    print(viol[:-1])
    print("Cost Delta:")
    print(viol[-1])

    # Compare with chordal SDP solve
    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(X_lr_new)
    axs[0].set_title("Rank Reduced SDP Soln")
    axs[1].matshow(X_lr)
    axs[1].set_title("Chordal SDP Soln")
    plt.show()

    # Difference in solution vectors
    s, U = np.linalg.eigh(X_lr)
    V_lr_c = U[:, [-1]] * np.sqrt(s[-1])
    print(f"Vector Solutions:")
    print(np.hstack([V, V_lr_c]))


if __name__ == "__main__":

    test_rank_reduction()
