import sys

from mosek.fusion import Domain, Expr, ObjectiveSense, Model, ProblemStatus
import numpy as np
import cvxpy as cp

from cert_tools.base_clique import BaseClique
from cert_tools.sdp_solvers import sdp_opts_dflt
from cert_tools.fusion_tools import mat_fusion, get_slice

CONSTRAIN_ALL_OVERLAP = False

TOL = 1e-5


def solve_oneshot_dual_slow(clique_list):
    """Implementation of range-space clique decomposition as in [Zheng 2020]."""
    from cert_tools.sdp_solvers import adjust_Q

    N = len(clique_list) + 1
    A_list = []
    for k, clique in enumerate(clique_list):
        clique.H = cp.Variable(clique.Q.shape, PSD=True)
        if k == 0:
            A_list += [clique.E.T @ A @ clique.E for A in clique.A_list]
        else:
            A_list += [clique.E.T @ clique.A_list[-1] @ clique.E]

    Q = cp.sum([clique.E.T @ clique.Q @ clique.E for clique in clique_list])
    Q_here, scale, offset = adjust_Q(Q)
    sigmas = cp.Variable(len(A_list))
    constraints = [
        cp.sum([clique.E.T @ clique.H @ clique.E for clique in clique_list])
        == Q_here + cp.sum([sigmas[k] * A_list[k] for k in range(len(A_list))])
    ]
    cprob = cp.Problem(cp.Maximize(-sigmas[0]), constraints)
    sdp_opts_dflt["verbose"] = True
    cprob.solve(solver="MOSEK", **sdp_opts_dflt)

    # H_k_list = [clique.H.value for clique in clique_list]
    X_k_list = constraints[0].dual_value
    sigma_dict = {i: sigma.value for i, sigma in enumerate(sigmas)}
    if not np.isinf(cprob.value):
        cost = cprob.value * scale + offset
        info = {"cost": cost, "sigma_dict": sigma_dict}
        info["success"] = True
    else:
        info = {"cost": np.inf, "sigma_dict": sigma_dict}
        info["success"] = False
    return X_k_list, info


def solve_oneshot_dual_cvxpy(clique_list, tol=TOL):
    """Implementation of range-space clique decomposition using auxiliary variables."""
    B_list_left = clique_list[0].get_B_list_left()
    B_list_right = clique_list[0].get_B_list_right()
    N = len(clique_list) + 1
    # raise ValueError("need to implement a fast dual version of this!")
    constraints = []
    sigmas = cp.Variable(N)
    rhos = cp.Variable(N - 1)
    for k, clique in enumerate(clique_list):
        if k == 0:
            z_var_left = None
            z_var_right = cp.Variable(len(B_list_right))

            s = sigmas[k : k + 2]
            A_list = clique.A_list[1:]
        elif k < N - 2:
            z_var_left = z_var_right
            z_var_right = cp.Variable(len(B_list_right))

            A_list = clique.A_list[2:]
            s = sigmas[k + 1 : k + 2]
        else:
            z_var_left = z_var_right  # previous right now becomes left.
            z_var_right = None

            A_list = clique.A_list[2:]
            s = sigmas[k + 1 : k + 2]

        clique.H = (
            clique.Q
            + rhos[k] * clique.A_list[0]
            + cp.sum([s[i] * A_list[i] for i in range(len(A_list))])
        )
        if z_var_left is not None:
            clique.H += cp.sum(
                [z_var_left[i] * B_list_left[i] for i in range(len(B_list_left))]
            )
        if z_var_right is not None:
            clique.H += cp.sum(
                [z_var_right[i] * B_list_right[i] for i in range(len(B_list_right))]
            )
        constraints += [clique.H >> 0]
    cprob = cp.Problem(cp.Maximize(-cp.sum(rhos)), constraints)
    data, *__ = cprob.get_problem_data(cp.SCS)
    sdp_opts_dflt["verbose"] = True
    cprob.solve(solver="MOSEK", **sdp_opts_dflt)

    X_k_list = [con.dual_value for con in constraints]
    # H_k_list = [clique.H.value for clique in clique_list]
    sigma_dict = {i: sigma.value for i, sigma in enumerate(sigmas)}
    if not np.isinf(cprob.value):
        cost = cprob.value
        info = {"cost": cost, "sigma_dict": sigma_dict}
        info["success"] = True
    else:
        info = {"cost": np.inf, "sigma_dict": sigma_dict}
        info["success"] = False
    return X_k_list, info


def solve_oneshot_primal_fusion(clique_list, verbose=False, tol=TOL):
    """
    clique_list is a list of objects inheriting from BaseClique.
    """
    assert isinstance(clique_list[0], BaseClique)

    X_dim = clique_list[0].X_dim
    N = len(clique_list)
    with Model("primal") as M:
        # creates (N x X_dim x X_dim) variable
        X = M.variable(Domain.inPSDCone(X_dim, N))

        # objective
        M.objective(
            ObjectiveSense.Minimize,
            Expr.add(
                [
                    Expr.dot(mat_fusion(clique_list[i].Q), get_slice(X, i))
                    for i in range(N)
                ]
            ),
        )

        # standard equality constraints
        for i, clique in enumerate(clique_list):
            for A, b in zip(clique.A_list, clique.b_list):
                A_fusion = mat_fusion(A)
                M.constraint(Expr.dot(A_fusion, get_slice(X, i)), Domain.equalsTo(b))

        # interlocking equality constraints
        for i in range(len(clique_list) - 1):
            for left_start, left_end, right_start, right_end in zip(
                clique_list[i].left_slice_start,
                clique_list[i].left_slice_end,
                clique_list[i].right_slice_start,
                clique_list[i].right_slice_end,
            ):
                X_left = X.slice([i] + left_start, [i + 1] + left_end)
                X_right = X.slice([i + 1] + right_start, [i + 2] + right_end)
                M.constraint(Expr.sub(X_left, X_right), Domain.equalsTo(0))

        M.setSolverParam("intpntCoTolDfeas", tol)  # default 1e-8
        M.setSolverParam("intpntCoTolPfeas", tol)  # default 1e-8
        M.setSolverParam("intpntCoTolMuRed", tol)  # default 1e-8
        if verbose:
            M.setLogHandler(sys.stdout)
        M.solve()
        if M.getProblemStatus() is ProblemStatus.Unknown:
            X_list_k = []
            info = {"success": False, "cost": np.inf}
        elif M.getProblemStatus() is ProblemStatus.PrimalAndDualFeasible:
            X_list_k = [
                np.reshape(get_slice(X, i).level(), (X_dim, X_dim)) for i in range(N)
            ]
            info = {"success": True, "cost": M.primalObjValue()}
        return X_list_k, info


def solve_oneshot_primal_cvxpy(clique_list, verbose=False, tol=TOL):
    constraints = []
    for k, clique in enumerate(clique_list):
        constraints += clique.get_constraints_cvxpy(clique.X)

    # add constraints for overlapping regions
    for k, clique in enumerate(clique_list):
        constraints += [clique.evaluate_F(clique.X, g=clique.g) == 0]

    cprob = cp.Problem(
        cp.Minimize(cp.sum([cp.trace(clique.Q @ clique.X) for clique in clique_list])),
        constraints,
    )

    sdp_opts_dflt["verbose"] = verbose
    cprob.solve(solver="MOSEK", **sdp_opts_dflt)

    X_k_list = [clique.X.value for clique in clique_list]
    sigma_dict = {
        i: constraint.dual_value
        for i, constraint in enumerate(constraints[-len(clique_list) :])
    }
    info = {"cost": cprob.value, "sigma_dict": sigma_dict}
    if not np.isinf(cprob.value):
        info["success"] = True
    else:
        info["success"] = False
    return X_k_list, info


def solve_oneshot(
    clique_list, use_primal=True, use_fusion=False, verbose=False, tol=TOL
):
    if use_primal:
        if use_fusion:
            return solve_oneshot_primal_fusion(clique_list, verbose=verbose, tol=tol)
        else:
            return solve_oneshot_primal_cvxpy(clique_list, verbose=verbose, tol=tol)
    else:
        if use_fusion:
            print("Warning: dual not implement for fusion, using cvxpy.")
        return solve_oneshot_dual_cvxpy(clique_list, verbose=verbose, tol=tol)
