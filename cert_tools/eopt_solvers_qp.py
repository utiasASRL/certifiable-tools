from copy import deepcopy

import cvxpy as cp
import numpy as np

from cert_tools.eig_tools import get_min_eigpairs
from cert_tools.sdp_solvers import sdp_opts_dflt
from cert_tools.eopt_solvers import get_cert_mat, get_grad_info

# tolerance for minimum eigevalue: mineig >= -TOL_EIG <=> A >= 0
TOL_EIG = 1e-10


# see Nocedal & Wright, Algorithm 3.1
backtrack_factor = 0.5  # rho (how much to decrase alpha)
backtrack_cutoff = 0.5  #  c (when to stop)
backtrack_start = 10.0  # starting value for alpha


def solve_d_from_indefinite_U(U, Q_1, A_vec, verbose=False):
    """
    Solve (17) from Overton 1992 to find a descent direciton d
    in case dual matrix U is indefinite.
    """
    m = A_vec.shape[1]
    n, t = Q_1.shape
    eig, vec = get_min_eigpairs(U, method="lanczos", k=1)

    d = cp.Variable(m)
    delta = cp.Variable()
    constraint = [
        cp.sum(
            [
                d[k] * Q_1.T @ A_vec[:, k].reshape((n, n), order="F") @ Q_1
                for k in range(m)
            ]
        )
        - delta * np.eye(t)
        == -vec @ vec.T
    ]
    prob = cp.Problem(cp.Minimize(1), constraints=constraint)
    prob.solve(solver="MOSEK", verbose=verbose > 2, **sdp_opts_dflt)

    success = d.value is not None
    info = {"msg": prob.status, "success": success, "delta": delta.value}
    return d.value, info


def solve_inner_QP(vecs, eigs, A_vec, t, rho, W, verbose=False, lmin=False):
    """
    Solve the direction-finding QP (Overton 1992, equations (24) - (27)).

    vecs and eigs are the eig-pairs at a current estimate, and t is the estimated multiplicity of the biggest one.

    """
    Q_1 = vecs[:, :t]
    lambdas = eigs[:t]
    eig_max = lambdas[0]
    n_eig = len(eigs)
    n = vecs.shape[0]
    m = A_vec.shape[1]

    d = cp.Variable(m)
    delta = cp.Variable()

    # create t(t+1)/2 constraints
    constraints = []
    rhs = np.diag(lambdas - eig_max)
    lhs = delta * np.eye(t) - cp.sum(
        [d[k] * Q_1.T @ A_vec[:, k].reshape((n, n), order="F") @ Q_1 for k in range(m)]
    )
    # constraints.append(rhs A_list== lhs)
    for i in range(t):
        for j in range(i, t):
            constraints.append(lhs[i, j] == rhs[i, j])
    constraints += [
        delta
        - cp.sum(
            [
                d[k]
                * vecs[:, i].T
                @ A_vec[:, k].reshape((n, n), order="F")
                @ vecs[:, i]
                for k in range(m)
            ]
        )
        >= eigs[i] - eig_max
        for i in range(t, n_eig)
    ]
    constraints += [cp.norm_inf(d) <= rho]

    prob = cp.Problem(cp.Minimize(delta + d.T @ W @ d), constraints=constraints)
    prob.solve(solver="MOSEK", verbose=verbose > 2, **sdp_opts_dflt)

    success = d.value is not None
    if success:
        # U = constraints[0].dual_value
        U = np.zeros((t, t))
        k = 0
        for i in range(t):
            for j in range(i, t):
                if i == j:
                    U[i, j] = constraints[k].dual_value
                else:
                    U[j, i] = constraints[k].dual_value / 2
                    U[i, j] = constraints[k].dual_value / 2
                k += 1
        eigs = np.linalg.eigvalsh(U)

        if np.all(eigs <= 0):
            U = -U
        d = d.value
        delta = delta.value
    else:
        print("Warning: didn't find feasible direction.")
        U = None
        d = None
        delta = None

    info = {"success": success, "msg": prob.status, "delta": delta}
    return U, d, info


def compute_current_W(vecs, eigs, A_vec, t, w):
    Q_1 = vecs[:, :t]
    n = Q_1.shape[0]
    m = A_vec.shape[1]
    U = cp.Variable((t, t), symmetric=True)
    constraints = [
        U >> 0,
    ]
    obj = cp.Minimize(
        cp.norm2(
            cp.trace(U)
            - 1
            + cp.sum(
                [
                    cp.trace(Q_1.T @ A_vec[:, k].reshape((n, n), order="F") @ Q_1 @ U)
                    for k in range(m)
                ]
            )
        )
    )
    prob = cp.Problem(obj, constraints=constraints)
    sol = prob.solve()

    U_est = U.value

    L_bar = np.diag(eigs[t:])
    Q_1_bar = vecs[:, t:]

    W = np.empty((m, m))
    for j in range(m):
        for k in range(j, m):
            G_jk = (
                2
                * Q_1.T
                @ A_vec[:, k].reshape((n, n), order="F")
                @ Q_1_bar
                @ np.diag([1 / (w - L_bar[i]) for i in range(L_bar.shape[0])])
                @ Q_1_bar.T
                @ A_vec[:, j].reshape((n, n), order="F")
                @ Q_1
            )
            if t == 1:
                W_jk = G_jk * U_est
            else:
                W_jk = np.trace(U_est @ G_jk)
            W[j, k] = W[k, j] = W_jk
    return W
