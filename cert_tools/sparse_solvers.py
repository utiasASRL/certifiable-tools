import itertools
import random
import sys
from time import time

import clarabel
import cvxpy as cp
import mosek.fusion.pythonic as fu
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from cert_tools.base_clique import BaseClique
from cert_tools.fusion_tools import get_slice, mat_fusion
from cert_tools.hom_qcqp import HomQCQP
from cert_tools.linalg_tools import smat
from cert_tools.sdp_solvers import (adjust_tol, adjust_tol_fusion,
                                    options_cvxpy, options_fusion)

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
    if isinstance(mat, sp.coo_matrix) or isinstance(mat, sp.coo_array):
        mat_fu = fu.Matrix.sparse(
            mat.shape[0], mat.shape[1], mat.row, mat.col, mat.data
        )
    elif isinstance(mat, sp.csc_matrix) or isinstance(mat, sp.csc_array):
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
    form="primal",
    reduce_constrs=None,
    verbose=False,
    tol=TOL,
    adjust=False,
    decomp_methods=dict(objective="split", constraint="greedy-cover"),
):
    """Solve decomposed SDP corresponding to input problem

    Args:
        prob (HomQCQP): Homogenous QCQP Problem
        verbose (bool, optional): If true, display solver output. Defaults to False.
        tol (float, optional): Tolerance for solver. Defaults to TOL.
        adjust (bool, optional): If true, adjust the cost matrix. Defaults to False.
    """
    if form == "primal":
        out = solve_dsdp_primal(
            problem=problem,
            reduce_constrs=reduce_constrs,
            verbose=verbose,
            tol=tol,
            adjust=adjust,
            decomp_methods=decomp_methods,
        )
    elif form == "dual":
        out = solve_dsdp_dual(
            problem=problem,
            verbose=verbose,
            tol=tol,
            adjust=adjust,
        )
    return out


def solve_feasibility_dsdp(
    problem: HomQCQP,
    x_cand: np.ndarray,
    verbose=False,
    tol=TOL,
    adjust=False,
    nu_0=None,
    soft_epsilon=False,
    eps_tol=1e-5,
    test_H_poly=None,
):
    """Solve decomposed SDP corresponding to input problem

    Args:
        problem (HomQCQP): Homogenous QCQP Problem
        x_cand (np.ndarray): Candidate solution that we try to certify
        verbose (bool, optional): If true, display solver output. Defaults to False.
        tol (float, optional): Tolerance for solver. Defaults to TOL.
        adjust (bool, optional): If true, adjust the cost matrix. Defaults to False.
    """
    A_h = PolyMatrix()
    A_h[problem.h, problem.h] = 1
    As = problem.As + [A_h]

    constraints = []

    cliques = problem.cliques
    cvars = [cp.Variable((c.size, c.size), PSD=True) for c in cliques]
    if test_H_poly is not None:
        assert isinstance(test_H_poly, PolyMatrix)
        test_H = test_H_poly.get_matrix(problem.var_sizes)
        test_cvars = [test_H_poly.get_matrix(c.var_sizes) for c in cliques]

    # LAGRANGE VARIABLES
    y = cp.Variable(len(As))
    if len(problem.Bs):
        u = cp.Variable(len(problem.Bs))
        constraints += [u >= 0]

    C_mat = problem.C.get_matrix(problem.var_sizes)

    cert_mat_list = [C_mat]
    # get constraint-multiplier products
    for i, A in enumerate(As):
        A_mat = A.get_matrix(problem.var_sizes)
        cert_mat_list.append(A_mat * y[i])

    for i, B in enumerate(problem.Bs):
        B_mat = B.get_matrix(problem.var_sizes)
        cert_mat_list.append(B_mat * u[i])
    H = cp.sum(cert_mat_list)

    # AFFINE CONSTRAINTS:
    # H_ij - sum(Z_k)_ij = C_ij + sum(Ai*y_i)_ij - sum(Z_k)_ij = 0
    # Get a list of edges in the aggregate sparsity pattern (including main diagonal)
    edges = [e.tuple for e in problem.asg.es]
    edges += [(v.index, v.index) for v in problem.asg.vs]

    # Generate one matrix constraint per edge. This links the cliques
    var_lookup_dict = {}
    for edge_id in edges:
        # Get variables in edge from graph
        var0 = problem.asg.vs["name"][edge_id[0]]
        var1 = problem.asg.vs["name"][edge_id[1]]

        # Get component of certificate matrix
        row_inds = problem._get_indices(var0)
        col_inds = problem._get_indices(var1)
        H_subblock = H[np.ix_(row_inds, col_inds)]
        if verbose:
            print(f"block {row_inds},{col_inds} of H", end=" ")

        # Find the cliques that are involved with these variables
        overlapping_cliques = [
            c for c in problem.cliques if (var0 in c.var_list) and (var1 in c.var_list)
        ]
        if len(overlapping_cliques):
            sum_list = [H_subblock]
            for clique in overlapping_cliques:
                unique_id = "".join(sorted([var0, var1]))
                if clique.index not in var_lookup_dict:
                    var_lookup_dict[clique.index] = []
                var_lookup_dict[clique.index].append(unique_id)

                if test_H_poly is not None:
                    test_sum_list = [test_H[np.ix_(row_inds, col_inds)]]

                row_inds = clique._get_indices(var0)
                col_inds = clique._get_indices(var1)
                if verbose:
                    print(f"minus {row_inds},{col_inds} of clique {clique}", end=" ")
                sum_list += [-cvars[clique.index][np.ix_(row_inds, col_inds)]]
                if test_H_poly is not None:
                    test_sum_list += [
                        -test_cvars[clique.index][np.ix_(row_inds, col_inds)]
                    ]

            # Add the list together
            if test_H_poly is not None:
                list_to_sum = np.concatenate(
                    [m.toarray()[:, :, None] for m in test_sum_list], axis=2
                )
                np.testing.assert_allclose(np.sum(list_to_sum, axis=2), 0.0)
            matsumvec = cp.sum(sum_list)
            constraints += [matsumvec == 0]
            if verbose:
                print("must equal zero.")
        else:
            raise ValueError(
                "inconsistency found: edge appears in graph but in none of the cliques."
            )

    # For each clique, we also need to make sure to constrain the parts that are not present in H.
    for clique in problem.cliques:
        # find all the variables that were not already constrained.
        for var0, var1 in itertools.combinations(clique.var_list, 2):
            unique_id = "".join(sorted([var0, var1]))
            if unique_id in var_lookup_dict[clique.index]:
                continue

            # Get component of certificate matrix
            row_inds = problem._get_indices(var0)
            col_inds = problem._get_indices(var1)
            if verbose:
                print(f"block {row_inds},{col_inds} of H", end=" ")

            sum_list = [H[np.ix_(row_inds, col_inds)]]

            if test_H_poly is not None:
                test_sum_list = [test_H[np.ix_(row_inds, col_inds)]]

            row_inds = clique._get_indices(var0)
            col_inds = clique._get_indices(var1)
            if verbose:
                print(f"minus {row_inds},{col_inds} of clique {clique}", end=" ")
            sum_list += [-cvars[clique.index][np.ix_(row_inds, col_inds)]]

            if test_H_poly is not None:
                test_sum_list += [-test_cvars[clique.index][np.ix_(row_inds, col_inds)]]

            # Add the list together
            if test_H_poly is not None:
                list_to_sum = np.concatenate(
                    [m.toarray()[:, :, None] for m in test_sum_list], axis=2
                )
                np.testing.assert_allclose(np.sum(list_to_sum, axis=2), 0.0)
            matsumvec = cp.sum(sum_list)
            constraints += [matsumvec == 0]
            if verbose:
                print("must equal zero.")

    if soft_epsilon:
        epsilon = cp.Variable()
        cost = cp.Minimize(epsilon)

        constraints += [H @ x_cand >= -epsilon]
        constraints += [H @ x_cand <= epsilon]
    else:
        cost = cp.Minimize(1.0)
        constraints += [H @ x_cand >= -eps_tol]
        constraints += [H @ x_cand <= eps_tol]

    if nu_0 is not None:
        constraints += [y[-1] == nu_0]
        # constraints += [y[-1] <= nu_0 + eps_tol]
        # constraints += [y[-1] >= nu_0 - eps_tol]

    # adjust tolerances
    adjust_tol(options_cvxpy, tol)

    prob = cp.Problem(cost, constraints)

    # Store information
    info = {"success": False, "cost": -np.inf, "msg": prob.status, "epsilon": None}

    prob.solve(accept_unknown=True)
    if prob.status == "optimal" or prob.status == "optimal_inaccurate":
        cost = prob.value
        clq_list = [cvar.value for cvar in cvars]
        # dual = [c.dual_variables[0].value for cvar in cvars for c in cvar.domain]
        ys = y.value
        us = u.value
        info["epsilon"] = prob.value if soft_epsilon else eps_tol
        info["success"] = True
        info["dual"] = "Not implemented"
        info["cost"] = cost
        info["yvals"] = ys[:-1]
        info["nu_0"] = ys[-1]
        info["mus"] = us
        info["messsage"] = prob.status
    else:
        print("Solve Failed - Mosek Status: " + str(prob.status))
        clq_list = []
    return clq_list, info


def solve_feasibility_dsdp_fusion(
    problem: HomQCQP,
    x_cand: np.ndarray,
    verbose=False,
    tol=TOL,
    adjust=False,
    nu_0=None,
    soft_epsilon=False,
    eps_tol=1e-5,
    test_H_poly=None,
):
    """Solve decomposed SDP corresponding to input problem

    Args:
        problem (HomQCQP): Homogenous QCQP Problem
        x_cand (np.ndarray): Candidate solution that we try to certify
        verbose (bool, optional): If true, display solver output. Defaults to False.
        tol (float, optional): Tolerance for solver. Defaults to TOL.
        adjust (bool, optional): If true, adjust the cost matrix. Defaults to False.
    """
    if adjust:
        raise NotImplementedError("adjust=True not implemented.")

    t0 = time()
    # List of constraints with homogenizing constraint.
    A_h = PolyMatrix()
    A_h[problem.h, problem.h] = 1
    As = problem.As + [A_h]

    # Define problem model
    M = fu.Model()

    # CLIQUE VARIABLES
    cliques = problem.cliques
    cvars = [M.variable(fu.Domain.inPSDCone(c.size)) for c in cliques]

    # LAGRANGE VARIABLES
    y = [M.variable(f"y{i}") for i in range(len(As))]
    if len(problem.Bs):
        u = [M.variable(f"u{j}") for j in range(len(problem.Bs))]
        [
            M.constraint(f"u{j}", u[j], fu.Domain.greaterThan(0.0))
            for j in range(len(problem.Bs))
        ]

    # CONSTRUCT CERTIFICATE
    if verbose:
        print("Building Certificate Matrix")

    C_mat = problem.C.get_matrix(problem.var_sizes)
    C_fusion = fu.Expr.constTerm(sparse_to_fusion(C_mat))

    cert_mat_list = [C_fusion]

    # get constraint-multiplier products
    for i, A in enumerate(As):
        A_mat = A.get_matrix(problem.var_sizes)
        A_fusion = sparse_to_fusion(A_mat)
        cert_mat_list.append(fu.Expr.mul(A_fusion, y[i]))

    for i, B in enumerate(problem.Bs):
        B_mat = B.get_matrix(problem.var_sizes)
        B_fusion = sparse_to_fusion(B_mat)
        cert_mat_list.append(fu.Expr.mul(B_fusion, u[i]))
    # sum into certificate
    H = fu.Expr.add(cert_mat_list)

    # AFFINE CONSTRAINTS:
    # H_ij - sum(Z_k)_ij = C_ij + sum(Ai*y_i)_ij - sum(Z_k)_ij = 0
    # Get a list of edges in the aggregate sparsity pattern (including main diagonal)
    if verbose:
        print("Generating Affine Constraints")
    edges = [e.tuple for e in problem.asg.es]
    edges += [(v.index, v.index) for v in problem.asg.vs]

    # Generate one matrix constraint per edge. This links the cliques
    var_lookup_dict = {}
    for edge_id in edges:
        # Get variables in edge from graph
        var0 = problem.asg.vs["name"][edge_id[0]]
        var1 = problem.asg.vs["name"][edge_id[1]]
        # Get component of certificate matrix
        row_inds = problem._get_indices(var0)
        col_inds = problem._get_indices(var1)
        inds = get_block_inds(row_inds, col_inds, var0 == var1)
        H_subblock = H.pick(inds)

        # Find the cliques that are involved with these variables
        overlapping_cliques = [
            c for c in problem.cliques if (var0 in c.var_list) and (var1 in c.var_list)
        ]
        if len(overlapping_cliques):
            sum_list = [H_subblock]
            for clique in overlapping_cliques:
                unique_id = "".join(sorted([var0, var1]))
                if clique.index not in var_lookup_dict:
                    var_lookup_dict[clique.index] = []
                var_lookup_dict[clique.index].append(unique_id)

                row_inds = clique._get_indices(var0)
                col_inds = clique._get_indices(var1)
                inds = get_block_inds(row_inds, col_inds, var0 == var1)
                sum_list += [-cvars[clique.index].pick(inds)]

            # Add the list together
            matsumvec = fu.Expr.add(sum_list)
            M.constraint(f"e_{var0}_{var1}", matsumvec, fu.Domain.equalsTo(0.0))

    # For each clique, we also need to make sure to constrain the parts that are not present in H.
    for clique in problem.cliques:
        # find all the variables that were not already constrained.
        for var0, var1 in itertools.combinations(clique.var_list, 2):
            unique_id = "".join(sorted([var0, var1]))
            if unique_id in var_lookup_dict[clique.index]:
                print("skipping", unique_id)
                continue

            # Get component of certificate matrix
            row_inds = problem._get_indices(var0)
            col_inds = problem._get_indices(var1)
            if verbose:
                print(f"block {row_inds},{col_inds} of H", end=" ")
            inds = get_block_inds(row_inds, col_inds, var0 == var1)
            H_subblock = H.pick(inds)

            row_inds = clique._get_indices(var0)
            col_inds = clique._get_indices(var1)
            if verbose:
                print(f"minus {row_inds},{col_inds} of clique {clique}", end=" ")
            inds = get_block_inds(row_inds, col_inds, var0 == var1)
            sum_list += [-cvars[clique.index].pick(inds)]

            # Add the list together
            matsumvec = fu.Expr.add(sum_list)
            M.constraint(f"e_{var0}_{var1}", matsumvec, fu.Domain.equalsTo(0.0))

    expression = fu.Expr.mul(H, x_cand)
    if verbose:
        print("Adding Objective")
    if soft_epsilon:
        epsilon = M.variable("epsilon")
        M.objective(fu.ObjectiveSense.Minimize, epsilon)

        expression1 = fu.Expr.mul(H, x_cand) - fu.Expr.vstack([epsilon] * len(x_cand))
        M.constraint(f"Hx_lt", expression1, fu.Domain.lessThan(0))
        expression2 = fu.Expr.mul(H, x_cand) + fu.Expr.vstack([epsilon] * len(x_cand))
        M.constraint(f"Hx_gt", expression2, fu.Domain.greaterThan(0))
    else:
        M.objective(fu.ObjectiveSense.Minimize, 1.0)
        M.constraint(f"Hx_lt", expression, fu.Domain.lessThan([eps_tol] * len(x_cand)))
        M.constraint(
            f"Hx_gt", expression, fu.Domain.greaterThan([-eps_tol] * len(x_cand))
        )

    if nu_0 is not None:
        M.constraint(f"nu_0_lt", y[-1], fu.Domain.equalsTo(nu_0))
        # M.constraint(f"nu_0_lt", y[-1], fu.Domain.lessThan(nu_0 + eps_tol))
        # M.constraint(f"nu_0_gt", y[-1], fu.Domain.greaterThan(nu_0 - eps_tol))

    # SOLVE
    # M.setSolverParam("intpntSolveForm", "dual")
    M.setSolverParam("intpntSolveForm", "primal")
    # record problem
    if verbose:
        M.writeTask("problem_dump_dual.ptf")
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
    t1 = time()
    M.solve()
    t2 = time()

    # Store information
    info = {
        "success": False,
        "cost": -np.inf,
        "runtime": t2 - t1,
        "preprocess_time": t1 - t0,
        "msg": str(M.getProblemStatus()),
    }

    # EXTRACT SOLN
    status = M.getProblemStatus()
    if status == fu.ProblemStatus.PrimalAndDualFeasible:
        # Get MOSEK cost
        cost = M.primalObjValue()
        clq_list = [cvar.level().reshape(cvar.shape) for cvar in cvars]
        yvals = [y_i.level()[0] for y_i in y]
        info["success"] = True
        info["dual"] = [cvar.dual().reshape(cvar.shape) for cvar in cvars]
        info["cost"] = cost
        info["yvals"] = yvals[:-1]
        info["nu_0"] = yvals[-1]
        info["mus"] = [u_i.level()[0] for u_i in u]
    else:
        print("Solve Failed - Mosek Status: " + str(status))
        clq_list = []
    return clq_list, info


def solve_dsdp_dual(
    problem: HomQCQP,
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
    T0 = time()
    # List of constraints with homogenizing constraint.
    A_h = PolyMatrix()
    A_h[problem.h, problem.h] = 1
    As = problem.As + [A_h]

    # Define problem model
    M = fu.Model()

    # CLIQUE VARIABLES
    cliques = problem.cliques
    cvars = [M.variable(fu.Domain.inPSDCone(c.size)) for c in cliques]

    # LAGRANGE VARIABLES
    y = [M.variable(f"y{i}") for i in range(len(As))]

    # OBJECTIVE
    if verbose:
        print("Adding Objective")
    M.objective(fu.ObjectiveSense.Minimize, y[-1])

    # CONSTRUCT CERTIFICATE
    if verbose:
        print("Building Certificate Matrix")
    cert_mat_list = []
    # get constant cost matrix
    C_mat = problem.C.get_matrix(problem.var_sizes)
    C_fusion = fu.Expr.constTerm(sparse_to_fusion(C_mat))
    cert_mat_list.append(C_fusion)
    # get constraint-multiplier products
    for i, A in enumerate(As):
        A_mat = A.get_matrix(problem.var_sizes)
        A_fusion = sparse_to_fusion(A_mat)
        cert_mat_list.append(fu.Expr.mul(A_fusion, y[i]))
    # sum into certificate
    H = fu.Expr.add(cert_mat_list)

    # AFFINE CONSTRAINTS:
    # H_ij - sum(Z_k)_ij = C_ij + sum(Ai*y_i)_ij - sum(Z_k)_ij = 0
    # Get a list of edges in the aggregate sparsity pattern (including main diagonal)
    if verbose:
        print("Generating Affine Constraints")
    edges = [e.tuple for e in problem.asg.es]
    edges += [(v.index, v.index) for v in problem.asg.vs]

    # Generate one matrix constraint per edge. This links the cliques to the
    for edge_id in edges:
        # Get variables in edge from graph
        var0 = problem.asg.vs["name"][edge_id[0]]
        var1 = problem.asg.vs["name"][edge_id[1]]
        # Get component of certificate matrix
        row_inds = problem._get_indices(var0)
        col_inds = problem._get_indices(var1)
        inds = get_block_inds(row_inds, col_inds, var0 == var1)
        sum_list = [H.pick(inds)]
        # Find the cliques that are involved with these variables
        clique_inds = problem.var_clique_map[var0] & problem.var_clique_map[var1]
        cliques = [problem.cliques[i] for i in clique_inds]
        # get components of clique variables
        for clique in cliques:
            if var0 in clique.var_list and var1 in clique.var_list:
                row_inds = clique._get_indices(var0)
                col_inds = clique._get_indices(var1)
                inds = get_block_inds(row_inds, col_inds, var0 == var1)
                sum_list.append(-cvars[clique.index].pick(inds))
        # Add the list together
        matsumvec = fu.Expr.add(sum_list)
        M.constraint(f"e_{var0}_{var1}", matsumvec, fu.Domain.equalsTo(0.0))

    # SOLVE
    M.setSolverParam("intpntSolveForm", "dual")
    # record problem
    if verbose:
        M.writeTask("problem_dump_dual.ptf")
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
    T1 = time()
    M.solve()
    T2 = time()

    # Store information
    info = {
        "success": False,
        "cost": -np.inf,
        "runtime": T2 - T1,
        "preprocess_time": T1 - T0,
        "msg": str(M.getProblemStatus()),
    }

    # EXTRACT SOLN
    status = M.getProblemStatus()
    if status == fu.ProblemStatus.PrimalAndDualFeasible:
        # Get MOSEK cost
        cost = M.primalObjValue()
        clq_list = [cvar.dual().reshape(cvar.shape) for cvar in cvars]
        dual = [cvar.level().reshape(cvar.shape) for cvar in cvars]
        mults = [y_i.level()[0] for y_i in y]
        info["success"] = True
        info["dual"] = dual
        info["cost"] = cost
        info["mults"] = mults
    else:
        print("Solve Failed - Mosek Status: " + str(status))
    return clq_list, info


def get_block_inds(row_inds, col_inds, triu=False):
    """Helper function for getting a grid of indices based on row and column indices. Only selects upper triangle if triu is set to true"""
    inds = []
    for row in range(len(row_inds)):
        if triu:
            colstart = row
        else:
            colstart = 0
        for col in range(colstart, len(col_inds)):
            inds.append([row_inds[row], col_inds[col]])
    return np.array(inds)


def solve_dsdp_primal(
    problem: HomQCQP,
    reduce_constrs=None,
    verbose=False,
    tol=TOL,
    adjust=False,
    decomp_methods=dict(objective="split", constraint="greedy-cover"),
):
    """Solve decomposed SDP corresponding to input problem

    Args:
        prob (HomQCQP): Homogenous QCQP Problem
        verbose (bool, optional): If true, display solver output. Defaults to False.
        tol (float, optional): Tolerance for solver. Defaults to TOL.
        adjust (bool, optional): If true, adjust the cost matrix. Defaults to False.
    """
    T0 = time()
    # Define problem model
    M = fu.Model()

    # CLIQUE VARIABLES
    cliques = problem.cliques
    cvars = [M.variable(fu.Domain.inPSDCone(c.size)) for c in cliques]

    def get_decomp_fusion_expr(pmat_in, decomp_method="split"):
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
    obj_expr = get_decomp_fusion_expr(
        problem.C, decomp_method=decomp_methods["objective"]
    )
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

    # EQUALITY CONSTRAINTS
    if verbose:
        print("Adding affine equality constraints")
    for iCnstr, A in enumerate(problem.As):
        constr_expr = get_decomp_fusion_expr(
            A, decomp_method=decomp_methods["constraint"]
        )
        M.constraint(
            "eq_" + str(iCnstr),
            constr_expr,
            fu.Domain.equalsTo(0.0),
        )

    # INEQUALITY CONSTRAINTS
    if verbose:
        print("Adding affine inequality constraints")
    for iCnstr, B in enumerate(problem.Bs):
        constr_expr = get_decomp_fusion_expr(
            B, decomp_method=decomp_methods["constraint"]
        )
        M.constraint(
            "ineq_" + str(iCnstr),
            constr_expr,
            fu.Domain.lessThan(0.0),
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
    T1 = time()
    M.solve()
    T2 = time()

    # Store information
    info = {
        "success": False,
        "cost": -np.inf,
        "runtime": T2 - T1,
        "preprocess_time": T1 - T0,
        "msg": str(M.getProblemStatus()),
    }

    # EXTRACT SOLN
    status = M.getProblemStatus()
    if status == fu.ProblemStatus.PrimalAndDualFeasible:
        # Get MOSEK cost
        cost = M.primalObjValue()
        clq_list = [cvar.level().reshape(cvar.shape) for cvar in cvars]
        dual = [cvar.dual().reshape(cvar.shape) for cvar in cvars]
        info["success"] = True
        info["dual"] = dual
        info["cost"] = cost
    else:
        print("Solve Failed - Mosek Status: " + str(status))

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
