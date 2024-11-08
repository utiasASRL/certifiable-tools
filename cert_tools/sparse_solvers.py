import itertools
import random
import sys
from time import time

import clarabel
import cvxpy as cp
import matplotlib.pyplot as plt
import mosek.fusion.pythonic as fu
import numpy as np
import scipy.sparse as sp
from igraph import Graph
from poly_matrix import PolyMatrix

from cert_tools.base_clique import BaseClique
from cert_tools.fusion_tools import get_slice, mat_fusion
from cert_tools.hom_qcqp import HomQCQP
from cert_tools.linalg_tools import smat, svec
from cert_tools.sdp_solvers import (
    adjust_tol,
    adjust_tol_fusion,
    options_cvxpy,
    options_fusion,
)

CONSTRAIN_ALL_OVERLAP = False

TOL = 1e-5


def solve_oneshot_dual_slow(clique_list, tol=TOL):
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
    options_cvxpy["verbose"] = True
    adjust_tol(options_cvxpy, tol)
    options_cvxpy["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] = tol
    cprob.solve(solver="MOSEK", accept_unknown=True, **options_cvxpy)

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


def solve_oneshot_dual_cvxpy(clique_list, tol=TOL, verbose=False, adjust=False):
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

    # data, *__ = cprob.get_problem_data(cp.SCS)

    options_cvxpy["verbose"] = verbose
    adjust_tol(options_cvxpy, tol)
    options_cvxpy["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] = tol
    cprob.solve(solver="MOSEK", accept_unknown=True, **options_cvxpy)

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


def sparse_to_fusion(mat: sp.coo_array):
    if isinstance(mat, sp.coo_matrix):
        mat_fu = fu.Matrix.sparse(
            mat.shape[0], mat.shape[1], mat.row, mat.col, mat.data
        )
    elif isinstance(mat, sp.csc_matrix):
        rows, cols = mat.nonzero()
        mat_fu = fu.Matrix.sparse(mat.shape[0], mat.shape[1], rows, cols, mat.data)
    else:
        raise ValueError("Matrix type not supported")
    return mat_fu


def solve_clarabel(problem: HomQCQP, use_decomp=False):
    """Use Clarabel to solve Homogenized SDP"""
    # Get problem data
    P, q, A, b = problem.get_standard_form()
    A = sp.csc_matrix(A)
    # Define cones
    cones = [clarabel.PSDTriangleConeT(problem.dim)]
    # settings
    settings = clarabel.DefaultSettings()
    # loosen tolerances
    tol = 1e-8
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol
    settings.tol_feas = tol
    settings.tol_infeas_abs = tol
    settings.tol_infeas_rel = tol
    settings.tol_ktratio = tol * 1e2

    # set up problem
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    # solve
    solution = solver.solve()
    # retrieve solution
    X = smat(solution.z)
    return X


def solve_dsdp(
    problem: HomQCQP,
    decomp_method="split",
    reduce_constrs=None,
    verbose=False,
    tol=TOL,
    adjust=False,
):
    """Solve decomposed SDP corresponding to input problem

    Args:
        prob (HomQCQP): Homogenous QCQP Problem
        verbose (bool, optional): If true, display solver output. Defaults to False.
        tol (float, optional): Tolerance for solver. Defaults to TOL.
        adjust (bool, optional): If true, adjust the cost matrix. Defaults to False.
    """

    # Define problem model
    M = fu.Model()

    # CLIQUE VARIABLES
    cliques = problem.cliques
    cvars = [M.variable(fu.Domain.inPSDCone(c.size)) for c in cliques]

    def get_decomp_fusion_expr(pmat_in):
        """decompose PolyMatrix and convert to fusion expression"""
        # decompose matrix
        mat_decomp = problem.decompose_matrix(pmat_in, decomp_method)
        # add clique components to fusion expression
        expr_sum_list = []
        for k, pmat in mat_decomp.items():
            clique = cliques[k]
            mat_k = pmat.get_matrix(variables=clique.var_sizes)
            mat_k_fusion = sparse_to_fusion(mat_k)
            expr_sum_list.append(fu.Expr.dot(mat_k_fusion, cvars[k]))
        expr = fu.Expr.add(expr_sum_list)
        return expr

    # OBJECTIVE
    if verbose:
        print("Adding Objective")
    obj_expr = get_decomp_fusion_expr(problem.C)
    M.objective(fu.ObjectiveSense.Minimize, obj_expr)

    # HOMOGENIZING CONSTRAINT
    A_h = PolyMatrix()
    A_h[problem.h, problem.h] = 1
    constr_expr_h = get_decomp_fusion_expr(A_h)
    M.constraint(
        "homog",
        constr_expr_h,
        fu.Domain.equalsTo(1.0),
    )

    # AFFINE CONSTRAINTS
    if verbose:
        print("Adding Affine Constraints")
    for iCnstr, A in enumerate(problem.As):
        constr_expr = get_decomp_fusion_expr(A)
        M.constraint(
            "c_" + str(iCnstr),
            constr_expr,
            fu.Domain.equalsTo(0.0),
        )

    # CLIQUE CONSISTENCY EQUALITIES
    if verbose:
        print("Generating overlap consistency constraints")
    clq_constrs = problem.get_consistency_constraints()
    # TEST reduce number of clique
    if reduce_constrs is not None:
        n_constrs = int(reduce_constrs * len(clq_constrs))
        clq_constrs = random.sample(clq_constrs, n_constrs)
    if verbose:
        print("Adding overlap consistency constraints to problem")
    cnt = 0
    for k, l, A_k, A_l in clq_constrs:
        # Convert sparse array to fusion sparse matrix
        A_k_fusion = sparse_to_fusion(A_k)
        A_l_fusion = sparse_to_fusion(A_l)
        # Create constraint
        expr = fu.Expr.dot(A_k_fusion, cvars[k]) + fu.Expr.dot(A_l_fusion, cvars[l])
        M.constraint(
            "ovrlap_" + str(k) + "_" + str(l) + "_" + str(cnt),
            expr,
            fu.Domain.equalsTo(0.0),
        )
        cnt += 1

    # SOLVE
    if verbose:
        print("Starting problem solve")
    M.setSolverParam("intpntSolveForm", "dual")
    # record problem
    if verbose:
        M.writeTask("problem_dump.ptf")
        print("Starting Solve")
    # adjust tolerances
    adjust_tol_fusion(options_fusion, tol)
    options_fusion["intpntCoTolRelGap"] = tol
    for key, val in options_fusion.items():
        M.setSolverParam(key, val)  # default 1e-8
    if verbose:
        M.setLogHandler(sys.stdout)
    else:
        f = open("mosek_output.tmp", "a+")
        M.setLogHandler(f)

    M.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
    T0 = time()
    M.solve()
    T1 = time()

    # EXTRACT SOLN
    if M.getProblemStatus() in [
        fu.ProblemStatus.PrimalAndDualFeasible,
        fu.ProblemStatus.Unknown,
    ]:
        # Get MOSEK cost
        cost = M.primalObjValue()
        if cost < 0:
            print("cost is negative! sanity check:")
        clq_list = [cvar.level().reshape(cvar.shape) for cvar in cvars]
        info = {
            "success": True,
            "cost": cost,
            "time": T1 - T0,
            "msg": M.getProblemStatus(),
        }
    elif M.getProblemStatus() is fu.ProblemStatus.DualInfeasible:
        clq_list = []
        info = {
            "success": False,
            "cost": -np.inf,
            "time": T1 - T0,
            "msg": "dual infeasible",
        }
    else:
        print("Unknown status:", M.getProblemStatus())
        clq_list = []
        info = {
            "success": False,
            "cost": -np.inf,
            "time": T1 - T0,
            "msg": M.getProblemStatus(),
        }
    return clq_list, info


def print_tuples(rows, cols, vals):
    for i in range(len(rows)):
        print(f"({rows[i]},{cols[i]},{vals[i]})")


def solve_oneshot_primal_fusion(junction_tree, verbose=False, tol=TOL, adjust=False):
    """
    junction_tree: a Graph structure that corresponds to the junction tree
    of the factor graph for the problem
    """
    if adjust:
        from cert_tools.sdp_solvers import adjust_Q

        raise ValueError("adjust_Q does not work when dealing with cliques")

    # Get list of clique objects
    clique_list = junction_tree.vs["clique_obj"]
    assert isinstance(clique_list[0], BaseClique)

    X_dim = clique_list[0].X_dim
    N = len(clique_list)
    with fu.Model("primal") as M:
        # creates (N x X_dim x X_dim) variable
        X = M.variable(fu.Domain.inPSDCone(X_dim, N))

        if adjust:
            Q_scale_offsets = [adjust_Q(c.Q) for c in clique_list]
        else:
            Q_scale_offsets = [(c.Q, 1.0, 0.0) for c in clique_list]

        # objective
        M.objective(
            fu.ObjectiveSense.Minimize,
            fu.Expr.add(
                [
                    fu.Expr.dot(mat_fusion(Q_scale_offsets[i][0]), get_slice(X, i))
                    for i in range(N)
                ]
            ),
        )

        # standard equality constraints
        A_0_constraints = []
        for i, clique in enumerate(clique_list):
            for A, b in zip(clique.A_list, clique.b_list):
                A_fusion = mat_fusion(A)
                con = M.constraint(
                    fu.Expr.dot(A_fusion, get_slice(X, i)), fu.Domain.equalsTo(b)
                )
                if b == 1:
                    A_0_constraints.append(con)

        # Loop through edges in the junction tree
        for iEdge, edge in enumerate(junction_tree.get_edgelist()):
            # Get cliques associated with edge
            cl = junction_tree.vs["clique_obj"][edge[0]]
            ck = junction_tree.vs["clique_obj"][edge[1]]
            for l in junction_tree.es["sepset"][iEdge]:
                for rl, rk in zip(cl.get_ranges(l), ck.get_ranges(l)):
                    # cl.X_var[rl[0], rl[1]] == ck.X[rk[0], rk[1]])
                    left_start = [rl[0][0], rl[1][0]]
                    left_end = [rl[0][-1] + 1, rl[1][-1] + 1]
                    right_start = [rk[0][0], rk[1][0]]
                    right_end = [rk[0][-1] + 1, rk[1][-1] + 1]
                    X_left = X.slice([cl.index] + left_start, [cl.index + 1] + left_end)
                    X_right = X.slice(
                        [ck.index] + right_start, [ck.index + 1] + right_end
                    )
                    M.constraint(fu.Expr.sub(X_left, X_right), fu.Domain.equalsTo(0))

                    if cl.X is not None and ck.X is not None:
                        np.testing.assert_allclose(
                            cl.X[
                                left_start[0] : left_end[0], left_start[1] : left_end[1]
                            ],
                            ck.X[
                                right_start[0] : right_end[0],
                                right_start[1] : right_end[1],
                            ],
                        )

        adjust_tol_fusion(options_fusion, tol)
        options_fusion["intpntCoTolRelGap"] = tol
        for key, val in options_fusion.items():
            M.setSolverParam(key, val)  # default 1e-8

        if verbose:
            M.setLogHandler(sys.stdout)
        else:
            f = open("mosek_output.tmp", "a+")
            M.setLogHandler(f)

        M.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
        M.solve()
        if M.getProblemStatus() in [
            fu.ProblemStatus.PrimalAndDualFeasible,
            fu.ProblemStatus.Unknown,
        ]:
            X_list_k = [
                np.reshape(get_slice(X, i).level(), (X_dim, X_dim)) for i in range(N)
            ]
            cost_raw = M.primalObjValue()
            if cost_raw < 0:
                print("cost is negative! sanity check:")
                for i, c in enumerate(clique_list):
                    print("mineig Q", np.linalg.eigvalsh(c.Q.toarray())[0])
                    print("mineig X", np.linalg.eigvalsh(X_list_k[i])[0])

            costs_per_clique = [con.dual()[0] for con in A_0_constraints]
            cost_test = sum(costs_per_clique)
            if abs(cost_test) > 1e-8:
                rel_err = abs((cost_raw - cost_test) / cost_test)
                assert rel_err < 1e-1, rel_err
            cost = sum(
                costs_per_clique[i] * Q_scale_offsets[i][1] + Q_scale_offsets[i][2]
                for i in range(N)
            )
            info = {"success": True, "cost": cost, "msg": M.getProblemStatus()}
        elif M.getProblemStatus() is fu.ProblemStatus.DualInfeasible:
            X_list_k = []
            info = {"success": False, "cost": -np.inf, "msg": "dual infeasible"}
        else:
            print("Unknown status:", M.getProblemStatus())
            X_list_k = []
            info = {"success": False, "cost": -np.inf, "msg": M.getProblemStatus()}
        return X_list_k, info


def solve_oneshot_primal_cvxpy(clique_list, verbose=False, tol=TOL):
    constraints = []
    for clique in clique_list:
        clique.X_var = cp.Variable((clique.X_dim, clique.X_dim), PSD=True)
        constraints += [
            cp.trace(A @ clique.X_var) == b
            for A, b in zip(clique.A_list, clique.b_list)
        ]

    # add constraints for overlapping regions
    for cl, ck in itertools.combinations(clique_list, 2):
        overlap = BaseClique.get_overlap(cl, ck, h=cl.hom)
        for l in overlap:
            for rl, rk in zip(cl.get_ranges(l), ck.get_ranges(l)):
                constraints.append(cl.X_var[rl[0], rl[1]] == ck.X_var[rk[0], rk[1]])
                if (cl.X is not None) and (ck.X is not None):
                    np.testing.assert_allclose(cl.X[rl[0], rl[1]], ck.X[rk[0], rk[1]])

    cprob = cp.Problem(
        cp.Minimize(
            cp.sum([cp.trace(clique.Q @ clique.X_var) for clique in clique_list])
        ),
        constraints,
    )

    options_cvxpy["verbose"] = verbose
    adjust_tol(options_cvxpy, tol)
    options_cvxpy["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] = tol
    cprob.solve(solver="MOSEK", accept_unknown=True, **options_cvxpy)

    X_k_list = [clique.X_var.value for clique in clique_list]
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
    junction_tree=None,
    clique_list=None,
    use_primal=True,
    use_fusion=False,
    verbose=False,
    tol=TOL,
):
    if not use_primal:
        print("Defaulting to primal because dual cliques not implemented yet.")
    if use_fusion:
        return solve_oneshot_primal_fusion(junction_tree, verbose=verbose, tol=tol)
    else:
        return solve_oneshot_primal_cvxpy(clique_list, verbose=verbose, tol=tol)
    # return solve_oneshot_dual_cvxpy(
    #        clique_list, verbose=verbose, tol=tol, adjust=adjust
    #    )
