from copy import deepcopy

import cvxpy as cp
import numpy as np

from cert_tools.eig_tools import get_min_eigpairs
from cert_tools.sdp_solvers import sdp_opts_dflt

# tolerance for minimum eigevalue: mineig >= -TOL_EIG <=> A >= 0
TOL_EIG = 1e-10


# see Nocedal & Wright, Algorithm 3.1
backtrack_factor = 0.5  # rho (how much to decrase alpha)
backtrack_cutoff = 0.5  #  c (when to stop)
backtrack_start = 10.0  # starting value for alpha


def get_subgradient(Q, Constraints, a):
    H_curr = Q + np.sum([ai * Ai for ai, (Ai, b) in zip(a, Constraints)])

    eig_vals, eig_vecs = get_min_eigpairs(H_curr)
    U = 1 / Q.shape[0] * np.eye(Q.shape[0])
    return eig_vecs @ U @ eig_vecs.T


def get_max_multiplicity(eigs, tau):
    "See equations (22), Overton 1992."
    eig_max = eigs[0]
    assert eig_max == np.max(eigs), "eigenvalues not sorted!"

    diff = eig_max - eigs - tau * max(1, abs(eig_max))
    # t is where diff_t <= 0, diff_{t+1} >= 0
    valid_idx = np.argwhere(diff >= 0)
    if len(valid_idx):
        # t_p1 = int(valid_idx[-1])
        t_p1 = int(valid_idx[0])
    else:
        t_p1 = len(eigs)
    assert diff[t_p1 - 1] <= 0
    return t_p1


def get_min_multiplicity(eigs, tau):
    "See equations (22), Overton 1992."
    eig_min = eigs[0]
    assert eig_min == np.min(eigs), "eigenvalues not sorted!"

    diff = eigs - eig_min - tau * max(1, abs(eig_min))
    # t is where diff_t <= 0, diff_{t+1} >= 0
    valid_idx = np.argwhere(diff >= 0)
    if len(valid_idx):
        # t_p1 = int(valid_idx[-1])
        t_p1 = int(valid_idx[0])
    else:
        t_p1 = len(eigs)
    assert diff[t_p1 - 1] <= 0
    return t_p1


def solve_d_from_indefinite_U(U, Q_1, Constraints, verbose=False):
    """
    Solve (17) from Overton 1992 to find a descent direciton d
    in case dual matrix U is indefinite.
    """
    m = len(Constraints)
    t = Q_1.shape[1]
    eig, vec = get_min_eigpairs(U, method="lanczos", k=1)

    d = cp.Variable(m)
    delta = cp.Variable()
    constraint = [
        cp.sum([d[k] * Q_1.T @ Constraints[k][0] @ Q_1 for k in range(m)])
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
    n = len(eigs)
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
        for i in range(t, n)
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


def f_eopt(Q, A_vec, x, lmin=False):
    """Objective of E-OPT"""
    # below doesn't work for sparse matrices.
    # H = Q + np.sum([x_i * Ai[None] for x_i, Ai in zip(x, A_list)], axis=0)[0]
    H = deepcopy(Q)
    H += (A_vec @ x).reshape(Q.shape, order="F")
    if lmin:
        eigs, __ = get_min_eigpairs(H, k=1, method="lanczos")
        return eigs[0]
    else:
        eigs, __ = get_min_eigpairs(-H, k=1, method="lanczos")
        return -eigs[0]


def solve_eopt_qp_simple(Q, A_list, a_init=None, max_iters=500, gtol=1e-7):
    """Solve max_alpha sigma_min (Q + sum_i (alpha_i * A_i)), using the QP algorithm
    provided by Overton 1988.
    """
    if a_init is None:
        a_init = np.zeros(len(A_list))

    alpha = 1.0

    a = a_init
    i = 0
    while i <= max_iters:
        subgrad = get_subgradient(Q, A_list, a)

        if np.linalg.norm(subgrad) < gtol:
            msg = "Converged in gradient norm"
            success = True
            break

        a += alpha * subgrad
        i += 1
        if i == max_iters:
            msg = "Reached maximum iterations"
            success = False
    info = {"success": success, "msg": msg}
    return a, info


def solve_eopt_qp(
    Q,
    A_vec,
    A_eq=None,
    b_eq=None,
    xinit=None,
    max_iters=1000,
    gtol=1e-10,
    verbose=1,
    lmin=True,
    l_threshold=0,
    kwargs_eig={},
):
    """Solve E_OPT: min_x sigma_max (Q + sum_m (x_m * A_m)), using the QP algorithm
                                    ----------H(x)---------
    provided by Overton 1992 (adaptation of Overton 1988).

    :param A_vec: NxM matrix of vectorized constraints.

    :param lmin: solve instead max_x sigma_min (Q + sum_i (x_i * A_i))
    """
    m = A_vec.shape[1]
    n = Q.shape[0]

    if (A_eq is not None) or (b_eq is not None):
        raise NotImplementedError("Can't use equality constraints in QP yet.")

    if xinit is None:
        xinit = np.zeros(m)
    else:
        xinit = xinit.flatten()

    # parameter for numerical multiplicity calculation
    tau = 1e-5

    # trust region radios
    rho = 1.0

    # weight matrix for QP
    W = np.zeros((m, m))
    U = None

    k = min(n, 5)
    method = "direct" if k == n else "lanczos"

    i = 0
    x = xinit
    print("it \t alpha \t\t l_min")
    while i <= max_iters:
        H = deepcopy(Q)
        H += (A_vec @ x).reshape((n, n), order="F")

        if lmin:  # maximize the minimum eigenvalue
            eigs, vecs = get_min_eigpairs(H, k=k, method=method)
            t = get_min_multiplicity(eigs, tau)
        else:  # minimize the maximum eigenvalue
            eigs, vecs = get_min_eigpairs(-H, k=k, method=method)
            eigs = -eigs
            vecs = -vecs
            t = get_max_multiplicity(eigs, tau)

        if i == 0 and verbose > 0:
            print(f"start \t ------ \t {eigs[0]:1.4e}")

        Q_1 = vecs[:, :t]

        if t == 1:
            grad = np.concatenate(
                [Q_1.T @ A_i.reshape((n, n), order="F") @ Q_1 for A_i in A_vec.T]
            ).flatten()
            l_old = f_eopt(Q, A_vec, x, lmin=lmin)
            d = grad / np.linalg.norm(grad)

            # make sure that the gradient is valid
            if lmin:  # we want to go up (maximize lmin)
                assert f_eopt(Q, A_vec, x + d * 1e-8, lmin=lmin) > l_old
            else:  # we want to go down (minimize lmax)
                assert f_eopt(Q, A_vec, x + d * 1e-8, lmin=lmin) < l_old

            # backgracking, using Nocedal Algorithm 3.1
            alpha = backtrack_start
            while np.linalg.norm(alpha * d) > gtol:
                l_new = f_eopt(Q, A_vec, x + alpha * d, lmin=lmin)
                # trying to minimize lambda
                l_test = l_old + backtrack_cutoff * alpha * grad.T @ d
                if not lmin and l_new <= l_test:
                    break

                # trying to maximize l_min
                elif lmin and (l_new >= l_test):
                    break
                alpha *= backtrack_factor

            # if np.linalg.norm(alpha * d) <= gtol:
            #    msg = "Converged in stepsize"
            #    success = True
            #    break
            x += alpha * d
        else:
            # w = max(Q_1.T @ H @ Q_1)
            # W = compute_current_W(vecs, eigs, A_list, t, w)
            U, d, info = solve_inner_QP(
                vecs, eigs, A_vec, t, rho=rho, W=W, verbose=verbose
            )

            if not info["success"]:
                raise ValueError("coudn't find feasible direction d")

            eigs_U = np.linalg.eigvalsh(U)

            if eigs_U[0] >= -TOL_EIG:
                l_emp = eigs[0] + info["delta"]
                l_new = f_eopt(Q, A_vec, x + d)
                if abs(l_new - l_emp) > 1e-10:
                    print(
                        f"Expected lambda not equal to actual lambda! {l_new:.6e}, {l_emp:.6e}"
                    )

                if np.linalg.norm(d) < gtol:
                    msg = "Converged in stepsize"
                    success = True
                    break
                else:
                    x += d
            else:
                print("Warning: U not p.s.d.")
                d, info = solve_d_from_indefinite_U(U, Q_1, A_vec)
                l_new = f_eopt(Q, A_vec, x + d)
                if d is not None:
                    x += d
                else:
                    tau = 0.5 * tau
                    rho = 0.5 * rho

        if verbose > 0:
            print(f"it {i} \t {alpha:1.4f} \t {l_new:1.4e}")

        if (l_threshold is not None) and (l_new >= l_threshold):
            msg = "Found valid certificate"
            success = True
            break

        i += 1
        if i == max_iters:
            msg = "Reached maximum iterations"
            success = False
    info = {"success": success, "msg": msg, "U": U, "lambda": l_new}
    return x, info


def solve_eopt_backtrack(Q, A_vec, x_init=None, max_iters=10, gtol=1e-7):
    """Solve min_x sigma_max (Q + sum_i (x_i * A_i)), using the QP algorithm
                             ----------H(x)---------
    provided by Overton 1992 (adaptation of Overton 1988).
    """
    m = A_vec.shape[1]
    n = Q.shape[0]

    if x_init is None:
        x_init = np.zeros(m)

    # parameter for numerical multiplicity calculation
    tau = 1e-3

    # trust region radios
    rho = 100.0

    # weight matrix for QP
    W = np.zeros((m, m))

    x = x_init
    i = 0
    while i <= max_iters:
        H = Q + (A_vec @ x).reshape(Q.shape, order="F")
        k = min(n, 10)
        eigs, vecs = get_min_eigpairs(H, k=k)
        eigs = eigs[::-1]
        vecs = vecs[:, ::-1]
        t = get_min_multiplicity(eigs, tau)
        Q_1 = vecs[:, :t]

        w = max(Q_1.T @ H @ Q_1)
        W = compute_current_W(vecs, eigs, A_vec, t, w)

        U, d, info = solve_inner_QP(vecs, eigs, A_vec, t, rho=rho, W=W)

        eigs = np.linalg.eigvalsh(U)
        if eigs[0] >= -TOL_EIG:
            if np.linalg.norm(d) < gtol:
                msg = "Converged in stepsize"
                success = True
                break
            else:
                x += d
        else:
            print("Warning: U not p.s.d.")
            tau = 0.5 * tau
            rho = 0.5 * rho

        i += 1
        if i == max_iters:
            msg = "Reached maximum iterations"
            success = False
    info = {"success": success, "msg": msg}
    return x, info
