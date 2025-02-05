import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
from diffcp.cones import unvec_symm, vec_symm

from cert_tools.linalg_tools import get_nullspace
from cert_tools.sdp_solvers import solve_sdp_fusion


def rank_reduction(
    Constraints,
    X_hr,
    rank_tol=1e-6,
    null_tol=1e-6,
    eig_tol=1e-9,
    null_method="svd",
    max_iter=None,
    verbose=False,
):
    """Algorithm that searches for a low rank solution to the SDP problem, given an existing high rank solution.
    Based on the algorithm proposed in "Low-Rank Semidefinite Programming:Theory and Applications by Lemon et al.
    
    
    """
    # Get initial low rank factor
    V, r = get_low_rank_factor(X_hr, rank_tol)
    if verbose:
        print(f"Initial rank of solution: {r}")
    # Get constraint operator matrix
    Av = get_constraint_op(Constraints, V)

    # REDUCE RANK
    dim_null = 1
    n_iter = 0
    while dim_null > 0 and (max_iter is None or n_iter < max_iter):
        # Compute null space
        # NOTE: This could be made faster by just computing a single right singular vector in the null space. No need to compute the entire space.
        basis, info = get_nullspace(Av, method=null_method, tolerance=null_tol)
        dim_null = basis.shape[0]
        if dim_null == 0:
            if verbose:
                print("Null space has no dimension. Exiting.")
            break
        # Get nullspace vector corresponding to the lowest gain eigenvalue
        Delta = unvec_symm(basis[-1], dim=V.shape[1])
        # Compute Eigenspace of Delta
        lambdas, Q = np.linalg.eigh(Delta)
        # find max magnitude eigenvalue
        indmax = np.argmax(np.abs(lambdas))
        max_lambda = lambdas[indmax]
        # Compute reduced lambdas
        alpha = -1 / max_lambda
        lambdas_red = 1 + alpha * lambdas
        # Check which eigenvalues are still nonzero
        inds = lambdas_red > eig_tol
        # Get update matrix
        Q_tilde = Q[:, inds] * np.sqrt(lambdas_red[inds])
        # Update Nullspace matrix
        Av = update_constraint_op(Av, Q_tilde, dim=r)
        # Update Factor
        V = V @ Q_tilde
        r = V.shape[1]
        n_iter += 1

        if verbose:
            print(f"iter: {n_iter}, dim null: {dim_null}, rank: {r}")

    return V


def rank_reduction_logdet(
    Constraints,
    X_hr,
    rank_tol=1e-6,
    stop_tol=1e-12,
    delta=1e-6,
    max_iter=1,
    verbose=False,
):
    """Defines the rank reduction problem over the (low-dimensional) SDP solution space.
    Solves the problem with a trace or log-det heuristic

    """

    # Get low rank factor
    V, r = get_low_rank_factor(X_hr, rank_tol)
    # Get reduced constraints
    reduced_constraints = get_reduced_constraints(Constraints, V)

    # Iterate to low rank solution
    n_iter = 0
    delta_S_norm = np.inf
    S = np.eye(r)
    S = np.diag([0, 0, 0, 1.0])
    while n_iter < max_iter and delta_S_norm > stop_tol:
        # Compute gradient
        grad_logdet = np.linalg.inv(S + delta * np.eye(r))
        # Solve SDP
        S_new, info = solve_sdp_fusion(
            grad_logdet,
            reduced_constraints,
            verbose=False,
            tol=1e-8,
            primal=False,
            adjust=False,
        )
        assert info["success"], ValueError("Mosek failed")
        # Compute delta
        delta_S_norm = np.linalg.norm(S_new - S)
        # Update
        S = S_new
        n_iter += 1

        if verbose:
            _, rank_s = get_low_rank_factor(S)
            print(f"ITER {n_iter}, DELTA {delta_S_norm}, RANK {rank_s}")

    # Compute low rank factor of reduced system
    V_s, _ = get_low_rank_factor(S)
    # Compute final low rank factor
    Y = V @ V_s

    return Y


def get_low_rank_factor(X, rank_tol=1e-6):
    """Get the low rank factorization of PSD matrix X. Tolerance is relative"""
    # get eigenspace
    vals, vecs = np.linalg.eigh(X)
    # remove zero eigenspace
    val_max = np.max(vals)
    r = np.sum(vals > rank_tol * val_max)
    n = X.shape[0]
    V = vecs[:, (n - r) :] * np.sqrt(vals[(n - r) :])
    return V, r


def get_reduced_constraints(Constraints, V):
    reduced_constraints = []
    for A, b in Constraints:
        reduced_constraints.append((V.T @ A @ V, b))
    return reduced_constraints


def get_constraint_op(Constraints, V):
    """Function to compute the constraint operator whose nullspace characterizes the optimal solution."""
    Av = []
    for A, b in Constraints:
        Av.append(vec_symm(V.T @ A @ V))
    Av = np.stack(Av)
    return Av


def update_constraint_op(Av, Q_tilde, dim):
    """Update the nullspace matrix. Updating this way is cheaper than reproducing the matrix, because it is performed in the lower dimension."""
    Av_updated = []
    for i, row in enumerate(Av):
        A = unvec_symm(row, dim=dim)
        Av_updated.append(vec_symm(Q_tilde.T @ A @ Q_tilde))
    Av_updated = np.stack(Av_updated)
    return Av_updated
