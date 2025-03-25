import casadi as cas
import matplotlib.pyplot as plt
import mosek.fusion as mf
import numpy as np
import scipy.sparse as sp
import sparseqr as sqr
from diffcp.cones import unvec_symm, vec_symm
from scipy.linalg import cholesky, lstsq, solve
from scipy.sparse.linalg import svds

from cert_tools.linalg_tools import get_nullspace
from cert_tools.sdp_solvers import solve_sdp_fusion


def rank_reduction(
    Constraints,
    X_hr,
    rank_tol=1e-6,
    null_tol=1e-6,
    eig_tol=1e-9,
    null_method="svd",
    targ_rank=None,
    max_iter=None,
    verbose=False,
):
    """Algorithm that searches for a low rank solution to the SDP problem, given an existing high rank solution.
    Based on the algorithm proposed in "Low-Rank Semidefinite Programming:Theory and Applications by Lemon et al.


    """
    # Get initial low rank factor
    V = get_low_rank_factor(X_hr, rank_tol)
    r = V.shape[1]
    if verbose:
        print(f"Initial rank of solution: {r}")
    # Get constraint operator matrix
    Av = get_constraint_op(Constraints, V)

    # REDUCE RANK
    n_iter = 0
    while (max_iter is None or n_iter < max_iter) and (
        targ_rank is None or r > targ_rank
    ):
        # Compute null space
        vec, s_min = get_min_sing_vec(Av, method=null_method)
        if targ_rank is None and s_min > null_tol:
            if verbose:
                print("Null space has no dimension. Exiting.")
            break
        # Get basis vector corresponding to the lowest gain eigenvalue (closest to nullspace)
        Delta = unvec_symm(vec, dim=V.shape[1])
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
            print(f"iter: {n_iter}, min s-value: {s_min}, rank: {r}")

    return V


def rank_inflation(
    C,
    Constraints,
    X_lr,
    max_iter=100,
    alpha=0.25,
    beta=0.99,
    lr_approx=True,
    damp=1e-4,
    verbose=False,
    debug=False,
):
    """Inflate the rank of a SDP solution using an analytic center technique.
    Applies equality constrained optimization with objective equal to -log-det"""
    # Unzip constraint list
    As, bs = zip(*Constraints)
    A_list = list(As)
    b_list = list(bs)
    # Add a constraint on cost
    A_list.append(C)
    b_list.append(np.trace(C @ X_lr))
    b = np.array(b_list)[:, None]
    # Shapes
    m = len(A_list)
    n = A_list[0].shape[0]

    # Init
    X = X_lr + 1e-5 * np.eye(n)
    # X = np.eye(n)
    n_iter = 0
    C = np.zeros((m, m))
    d = np.zeros((m, 1))
    while n_iter < max_iter:
        # Compute AX products
        AX = [A @ X for A in A_list]
        if lr_approx:
            V = get_low_rank_factor(X)
            VAV = [V.T @ A @ V for A in A_list]
        # Construct linear system for multipliers
        for i in range(m):
            d[i] = np.trace(AX[i])
            for j in range(i, m):
                if lr_approx:
                    C[i, j] = -np.trace(VAV[i] @ VAV[j])
                else:
                    C[i, j] = -np.sum(AX[i].T * AX[j])
                if j > i:
                    C[j, i] = C[i, j]
        # Solve symmetric linear system
        # w = solve(C, b - 2 * d, assume_a="sym")
        res = lstsq(C + damp * np.eye(m), b - 2 * d)
        w = res[0]
        # Compute search direction
        del_X = X - X @ np.sum([A_list[i] * w[i, 0] for i in range(m)]) @ X
        decrement = np.linalg.norm(del_X)
        if debug:
            # full step should yield primal residual of zero.
            X_plus = X + del_X
            viol = [np.trace(A_list[i] @ X_plus) - b[i, 0] for i in range(m)]
            print(f"DEBUG: max violation: {np.max(np.abs(viol))}")
        # Backtracking linesearch
        L = cholesky(X)
        L_inv = solve(L, np.eye(n))
        # These eigenvalues determine our search length
        evals = np.linalg.eigvalsh(L_inv @ del_X @ L_inv)
        eval_min = np.min(evals)
        if eval_min < 0:
            t_init = beta * min(-1 / eval_min, 1)
        else:
            t_init = 1
        t = backtrack_linesearch(evals, alpha, beta, t_init)
        # Update
        X = X + t * del_X
        n_iter = n_iter + 1
        # print results
        if verbose:
            res_pri = np.linalg.norm(b - d)
            print(f"Iter {n_iter}:  Newt.decr: {decrement},   res_pri: {res_pri}")

    return X


def backtrack_linesearch(evals, alpha, beta, t_init):
    """Backtracking linesearch for analytic centering problem"""
    # logdetX = np.log(np.linalg.det(X))
    grad_f = -np.sum(evals)
    t = t_init

    bt_cond = True
    # func_0 = -logdetX
    func_0 = 0
    while bt_cond:
        # Compute objective at t
        # func_t = -logdetX - np.sum(np.log(1 + t * evals))
        func_t = -np.sum(np.log(1 + t * evals))
        assert func_t is not np.nan, ValueError(
            "Negative eigenvalue in backtrack computation"
        )
        # Check condition
        bt_cond = func_t > func_0 + alpha * t * grad_f
        # Update if necessary
        if bt_cond:
            t = beta * t

    return t


def rank_inflation_sdp(C, Constraints, X_lr, verbose=False):
    """Inflate the rank keeping cost and constraint values the same using a feasibility problem."""
    # Add cost constraint to list
    Constraints_cost = Constraints.copy()
    Constraints_cost.append((C, np.trace(C @ X_lr)))
    # Starting point
    eps = 1e-6
    X_init = X_lr + eps * np.eye(X_lr.shape[0])

    # Solve Feasibility problem
    options = dict()
    X_maximal, info = solve_sdp_fusion(
        Q=C * 0,
        Constraints=Constraints_cost,
        adjust=False,
        verbose=verbose,
        options=options,
        X_init=X_init.flatten(),
    )

    return X_maximal


def get_sparse_nullspace(A, tolerance=1e-9):
    # We "solve" a least squares problem to get the rank and permutations. This is the cheapest way to use sparse QR, since it does not require explicit construction of the Q matrix. We can't do this with qr function because the "just return R" option is not exposed.
    Z, R, E, rank = sqr.rz(A, np.zeros((A.shape[0], 1)), tolerance=tolerance)
    if not rank == A.shape[0]:
        raise ValueError("QR decomposition rank is incorrect")
    # Get nullspace
    R = sp.csc_array(R)
    R11, R12 = R[:rank, :rank], R[:rank, rank:]
    # [R11  R12]  @  [R11^-1 @ R12] = [R12 - R12] = 0
    # [0    0 ]       [    -I    ]    [0]
    N = sp.vstack(
        [sp.linalg.spsolve_triangular(R11, R12.todense()), -sp.eye(R12.shape[1])]
    )
    # Unpermute the basis
    basis = N[E, :]

    return basis


def rank_inflation_lmi(
    C,
    Constraints,
    X_lr,
    eps=1e-3,
    null_tol=1e-8,
    max_iter=100,
    verbose=False,
):
    """Inflate the rank using a projected gradient approach"""
    # # Get constraint operator
    A_bar = get_full_constraint_op(Constraints, C)
    # # Compute nullspace using sparse rank revealing QR
    F = get_sparse_nullspace(A_bar, tolerance=null_tol)

    # Initial decomposition
    V_0 = get_low_rank_factor(X_lr)
    n_iter = 0
    n, rank = V_0.shape
    V = V_0


def rank_inflation_pg(
    C,
    Constraints,
    X_lr,
    eps=1e-2,
    max_iter=100,
    alpha=0.25,
    beta=0.8,
    verbose=False,
):
    """Inflate the rank using a projected gradient approach"""
    # Get constraint operator
    A_bar = get_full_constraint_op(Constraints, C)

    # Initial decomposition
    V_0 = get_low_rank_factor(X_lr)
    # Iterate
    n_iter = 0
    n, rank = V_0.shape
    V = V_0
    while n_iter < max_iter:
        # GRADIENT COMPUTATION
        # Compute gradient of log det function using SMW trick
        D = eps * np.eye(rank) + V.T @ V
        DinvV = np.linalg.solve(D, V.T)
        Grad = 1 / eps * (np.eye(n) - V @ DinvV)
        # NULL SPACE PROJECTION
        grad = Grad.reshape((-1, 1))
        grad_proj = nullspace_projection(A_bar, grad)
        del_X = grad_proj.reshape((n, n))
        # LINESEARCH
        L = cholesky(V @ V.T + eps * np.eye(n))
        L_inv = solve(L, np.eye(n))
        # These eigenvalues determine our search length
        evals = np.linalg.eigvalsh(L_inv @ del_X @ L_inv)
        eval_min = np.min(evals)
        if eval_min < 0:
            t_init = beta * min(-1 / eval_min, 1)
        else:
            t_init = 1
        t = backtrack_linesearch(evals, alpha, beta, t_init)
        # Update and refactor
        X = V @ V.T + t * del_X
        V = get_low_rank_factor(X)
        r = V.shape[1]
        n_iter += 1
        if verbose:
            print(f"Iter {n_iter}, rank: {r}, alpha: {alpha}")


def get_full_constraint_op(Constraints, C):
    """Function to compute the constraint operator whose nullspace characterizes the optimal solution.
    Computes the full operator rather than a subspace."""
    A_bar = []
    for A, b in Constraints:
        A_bar.append(A.tocoo().reshape((1, -1)))
    A_bar.append(C.tocoo().reshape((1, -1)))
    A_bar = sp.vstack(A_bar)
    return A_bar


def nullspace_projection(A, x, method="direct"):
    """Solve the nullspace projection for a large sparse matrix. That is we want to find:
    x_p = (I - A^T (A A^T)^(-1) A) x = x - (A^+) A x

    We solve this by defining y = A x, then solving A z = y as a least squares problem
    """

    if method == "cg":
        Ax = A @ x
        # construct normal eq matrix
        AAt = A @ A.T
        z, info = sp.linalg.cg(AAt, Ax)
        # Projection
        x_proj = x - A.T @ z[:, None]
    elif method == "lsqr":
        Ax = A @ x
        output = sp.linalg.lsqr(A, Ax)
        z = output[0]
        x_proj = x - z[:, None]
    elif method == "direct":
        Ax = A @ x
        # construct normal eq matrix
        AAt = A @ A.T
        z = np.linalg.solve(AAt.toarray(), Ax)
        # Projection
        x_proj = x - A.T @ z
    return x_proj


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
    V = get_low_rank_factor(X_hr, rank_tol)
    r = V.shape[1]
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
            S_lr = get_low_rank_factor(S)
            rank_s = S_lr.shape[1]
            print(f"ITER {n_iter}, DELTA {delta_S_norm}, RANK {rank_s}")

    # Compute low rank factor of reduced system
    V_s = get_low_rank_factor(S)
    # Compute final low rank factor
    Y = V @ V_s

    return Y


def get_min_sing_vec(A, method="svd"):
    """Get vector associated with minimum singular value."""
    if method == "svds":
        # Get minimum singular vector
        # NOTE: This method is fraught with numerical issues, but should be faster than computing all of the singular values
        s_min, vec = svds(A, k=1, which="SM")

    elif method == "svd":
        # Get all singular vectors (descending order)
        U, S, Vh = np.linalg.svd(A)
        s_min = S[-1]
        vec = Vh[-1, :]
    else:
        raise ValueError("Singular vector method unknown")

    return vec, s_min


def get_low_rank_factor(X, rank_tol=1e-6, rank=None):
    """Get the low rank factorization of PSD matrix X. Tolerance is relative"""
    # get eigenspace
    vals, vecs = np.linalg.eigh(X)
    # remove zero eigenspace
    val_max = np.max(vals)
    if rank is None:
        rank = np.sum(vals > rank_tol * val_max)
    n = X.shape[0]
    V = vecs[:, (n - rank) :] * np.sqrt(vals[(n - rank) :])
    return V


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
