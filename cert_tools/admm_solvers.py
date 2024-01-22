import sys
from copy import deepcopy
from multiprocessing import Pipe, Process

import cvxpy as cp
import mosek.fusion as fu
import numpy as np
from cert_tools.admm_clique import ADMMClique
from cert_tools.fusion_tools import mat_fusion
from cert_tools.sdp_solvers import (
    adjust_Q,
    adjust_tol,
    adjust_tol_fusion,
    options_cvxpy,
    options_fusion,
)
from cert_tools.sparse_solvers import read_costs_from_mosek

EARLY_STOP = True
EARLY_STOP_MIN = 1e-3

RHO_START = 1e2

MAXITER = 1000

# See [Boyd 2010] for explanations of these.
MU_RHO = 2.0  # how much dual and primal residual may get unbalanced.
TAU_RHO = 2.0  # how much to change rho in each iteration.
EPS_ABS = 0.0  # set to 0 to use relative only
EPS_REL = 1e-10
INDIVIDUAL_RHO = False  # use different rho for each clique
VECTOR_RHO = False  # use different rho for each error term.
UPDATE_RHO = True  # if False, never update rho at all

# Stop ADMM if the last N_ADMM iterations don't have significant change in cost.
N_ADMM = 3

# Tolerance of inner SDP for ADMM
TOL_INNER = 1e-5


def initialize_z(clique_list, X0=None):
    """Initialize Z (consensus variable) based on contents of X0 (initial feasible points)"""
    for k, clique in enumerate(clique_list):
        if X0 is not None:
            clique.z_new = deepcopy(clique.F @ X0[k].flatten())
        else:
            clique.z_new = np.zeros(clique.F.shape[0])


def check_convergence(clique_list, primal_err, dual_err):
    """Check convergence using the criteria by [Boyd 2010]."""
    dim_primal = (len(clique_list) - 1) * clique_list[0].X_dim
    dim_dual = len(clique_list) * clique_list[0].X_dim
    eps_max = np.max(
        [
            np.max(np.hstack([clique.X_new.flatten(), clique.z_new]))
            for clique in clique_list[:-1]
        ]
    )
    eps_pri = np.sqrt(dim_primal) * EPS_ABS + EPS_REL * eps_max
    eps_dual = np.sqrt(dim_dual) * EPS_ABS + EPS_REL * np.linalg.norm(
        np.hstack([clique.sigmas for clique in clique_list])
    )
    return (primal_err < eps_pri) and (dual_err < eps_dual)


def update_rho(rho, dual_res, primal_res, mu=MU_RHO, tau=TAU_RHO):
    """Update rho as suggested by [Boyd 2010]."""
    assert tau >= 1.0
    assert mu >= 0
    if np.ndim(rho) > 0:
        rho[np.where(primal_res >= mu * dual_res)[0]] *= tau
        rho[np.where(dual_res >= mu * primal_res)[0]] /= tau
        return rho
    else:
        if primal_res >= mu * dual_res:
            return rho * tau
        elif dual_res >= mu * primal_res:
            return rho / tau
        else:
            return rho


def update_z(clique_list):
    """Average the overlapping areas of X_new for consensus (stored in Z_new)."""
    for i, c in enumerate(clique_list):
        this = c.X_new.flatten()
        left = clique_list[i - 1].X_new.flatten() if i > 0 else None
        right = clique_list[i + 1].X_new.flatten() if i < len(clique_list) - 1 else None

        c.z_prev = deepcopy(c.z_new)
        # for each Z, average over the neighboring cliques.
        if i == 0:
            c.z_new = 0.5 * (c.F_right @ this + c.F_left @ right)
        elif i == len(clique_list) - 1:
            c.z_new = 0.5 * (c.F_right @ left + c.F_left @ this)
        else:
            c.z_new = 0.5 * np.hstack(
                [
                    c.F_right @ left + c.F_left @ this,
                    c.F_right @ this + c.F_left @ right,
                ]
            )
        assert c.z_new.shape == c.z_prev.shape


def update_sigmas(clique_list):
    """Update sigmas (dual variables) and residual terms."""
    for clique in clique_list:
        assert isinstance(clique, ADMMClique)
        g = clique.z_new
        clique.primal_res_k = clique.F @ clique.X_new.flatten() - g
        clique.dual_res_k = clique.z_new - clique.z_prev
        clique.sigmas += clique.rho_k * clique.primal_res_k


# TODO(FD): this function is hideous. Can we simplify / remove it somehow?
def wrap_up(
    clique_list,
    cost_history,
    primal_err,
    dual_err,
    iter,
    early_stop,
    verbose=False,
):
    """Calculate and print statistics, check stopping criteria etc."""
    info = {}
    cost_original = cost_history[-1]
    if verbose:
        with np.printoptions(precision=2, suppress=True, threshold=5):
            if iter % 20 == 0:
                print("iter          prim. error         dual error       cost")
            print(
                f"{iter} \t {primal_err:2.4e} \t {dual_err:2.4e} \t {cost_original:5.5f}"
            )
            # print("rho:", [c.rho_k for c in clique_list])

    rel_diff = (
        np.max(np.abs(np.diff(cost_history[-N_ADMM:]))) / abs(cost_history[-1])
        if len(cost_history) >= N_ADMM
        else None
    )
    if check_convergence(clique_list, primal_err, dual_err):
        info["success"] = True
        info["msg"] = f"converged after {iter} iterations"
        info["stop"] = True
    elif early_stop and rel_diff and (rel_diff < EARLY_STOP_MIN):
        info["success"] = True
        info["msg"] = f"stopping after {iter} because cost didn't change enough"
        info["stop"] = True
    # All cliques did not solve in the last iteration.
    elif all([c.status < 0 for c in clique_list]):
        info["success"] = False
        info["msg"] = f"all problems infeasible after {iter} iterations"
        info["stop"] = True
    return info


def solve_inner_sdp_fusion(Q, Constraints, F, g, sigmas, rho, verbose=False, tol=1e-8):
    """Solve X update of ADMM using fusion."""
    with fu.Model("primal") as M:
        # creates (N x X_dim x X_dim) variable
        X = M.variable("X", fu.Domain.inPSDCone(Q.shape[0]))
        if F is not None:
            S = M.variable("S", fu.Domain.inPSDCone(F.shape[0] + 2))
            a = M.variable("a", fu.Domain.inPSDCone(1))

        # standard equality constraints
        for A, b in Constraints:
            M.constraint(fu.Expr.dot(mat_fusion(A), X), fu.Domain.equalsTo(b))

        # interlocking equality constraints
        Q_here, offset, scale = adjust_Q(Q, scale_method="fro")
        if F is not None:
            assert g is not None
            if F.shape[1] == Q.shape[0]:
                err = fu.Expr.sub(
                    fu.Expr.mul(F, X.slice([0, 0], [Q.shape[0], 1])),
                    fu.Matrix.dense(g.value[:, None]),
                )
            else:
                err = fu.Expr.sub(fu.Expr.mul(F, fu.Expr.flatten(X)), g)

            # doesn't work unforuntately:
            # Expr.mul(0.5 * rho, Expr.sum(Expr.mulElm(err, err))),
            M.objective(
                fu.ObjectiveSense.Minimize,
                fu.Expr.add(
                    [
                        fu.Expr.dot(mat_fusion(Q), X),
                        fu.Expr.sum(
                            fu.Expr.mul(fu.Matrix.dense(sigmas[None, :]), err)
                        ),  # sum is to go from [1,1] to scalar
                        fu.Expr.sum(a),
                    ]
                ),
            )
            M.constraint(fu.Expr.sub(S.index([0, 0]), a), fu.Domain.equalsTo(0.0))
            M.constraint(
                fu.Expr.sub(
                    S.slice([1, 1], [1 + F.shape[0], 1 + F.shape[0]]),
                    fu.Matrix.sparse(
                        F.shape[0],
                        F.shape[0],
                        range(F.shape[0]),
                        range(F.shape[0]),
                        [2 / rho] * F.shape[0],
                    ),
                ),
                fu.Domain.equalsTo(0.0),
            )
            M.constraint(
                fu.Expr.sub(S.slice([1, 0], [1 + F.shape[0], 1]), err),
                fu.Domain.equalsTo(0.0),
            )
        else:
            M.objective(fu.ObjectiveSense.Minimize, fu.Expr.dot(Q_here, X))

        adjust_tol_fusion(options_fusion, tol)
        options_fusion["intpntCoTolRelGap"] = tol * 10
        for key, val in options_fusion.items():
            M.setSolverParam(key, val)
        if verbose:
            M.setLogHandler(sys.stdout)
        else:
            f = open("mosek_output.tmp", "a+")
            M.setLogHandler(f)
        M.solve()
        if M.getProblemStatus() is fu.ProblemStatus.Unknown:
            cost = np.inf
            if not verbose:
                f.close()
                primal_value, dual_value = read_costs_from_mosek("mosek_output.tmp")
                if (abs(primal_value) - abs(dual_value)) / abs(primal_value) > 1e-2:
                    print("Warning: solution not good")
                cost = abs(primal_value)
            cost = cost * scale + offset
            info = {"success": False, "cost": cost, "msg": "UNKNOWN"}
            X = None
        elif M.getProblemStatus() is fu.ProblemStatus.PrimalAndDualFeasible:
            X = np.reshape(X.level(), Q.shape)
            cost = M.primalObjValue()
            cost = cost * scale + offset
            info = {"success": True, "cost": cost, "msg": "solved"}
    return X, info


def solve_inner_sdp(
    clique: ADMMClique, rho=None, verbose=False, use_fusion=True, tol=1e-8
):
    """Solve the inner SDP of the ADMM algorithm, similar to [Dall'Anese 2013]

    min <Q, X> + y'e(X) + rho/2*||e(X)||^2
    s.t. <Ai, X> = bi
         X >= 0

    where e(X) = F @ vec(X) - b
    """
    if use_fusion:
        Constraints = list(zip(clique.A_list, clique.b_list))
        return solve_inner_sdp_fusion(
            clique.Q,
            Constraints,
            clique.F,
            clique.g,
            clique.sigmas,
            rho,
            verbose,
            tol=tol,
        )
    else:
        objective = clique.get_objective_cvxpy(clique.X_var, rho)
        constraints = clique.get_constraints_cvxpy(clique.X_var)
        cprob = cp.Problem(objective, constraints)
        options_cvxpy["verbose"] = verbose
        adjust_tol(options_cvxpy, tol)
        options_cvxpy["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] = tol * 10
        try:
            cprob.solve(solver="MOSEK", **options_cvxpy)
            info = {
                "cost": float(cprob.value),
                "success": clique.X_var.value is not None,
            }
        except Exception as e:
            info = {"cost": np.inf, "success": False}
        return clique.X_var.value, info


def solve_alternating(
    clique_list,
    X0: list[np.ndarray] = None,
    sigmas: dict = None,
    rho_start: float = RHO_START,
    use_fusion: bool = True,
    early_stop: bool = EARLY_STOP,
    verbose: bool = False,
    maxiter=MAXITER,
    mu_rho=MU_RHO,
    tau_rho=TAU_RHO,
    individual_rho=INDIVIDUAL_RHO,
    vector_rho=VECTOR_RHO,
):
    """Use ADMM to solve decomposed SDP, but without using parallelism."""
    assert not (vector_rho and not individual_rho), "incompatible combination"
    if sigmas is not None:
        for k, sigma in sigmas.items():
            clique_list[k].sigmas = sigma

    for c in clique_list:
        if vector_rho:
            c.rho_k = np.full(c.F.shape[0], rho_start).astype(float)
        else:
            c.rho_k = rho_start

    # rho_k = rho_start
    info_here = {"success": False, "msg": "did not converge.", "stop": False}

    cost_history = []

    initialize_z(clique_list, X0)  # fill Z_new
    for iter in range(maxiter):
        cost_lagrangian = 0
        cost_original = 0

        # ADMM step 1: update X
        for k, clique in enumerate(clique_list):
            assert isinstance(clique, ADMMClique)
            # update g with solved value from previous iteration
            clique.g = clique_list[k].z_new
            assert clique.g is not None

            X, info = solve_inner_sdp(
                clique,
                clique.rho_k,
                verbose=False,
                use_fusion=use_fusion,
                tol=TOL_INNER,
            )
            cost = info["cost"]

            if X is not None:
                clique.X_new = deepcopy(X)
                clique.status = 1
                cost_original += float(np.trace(clique.Q @ clique.X_new))
            else:
                print(f"clique {k:02.0f} did not converge!!")
                clique.status = -1
            cost_lagrangian += cost

        # ADMM step 2: update Z
        update_z(clique_list)

        # ADMM step 3: update Lagrange multipliers
        update_sigmas(clique_list)
        primal_err = np.linalg.norm(np.hstack([c.primal_res_k for c in clique_list]))
        dual_err = np.linalg.norm(np.hstack([c.dual_res_k for c in clique_list]))

        # update rho
        if UPDATE_RHO:
            for c in clique_list:
                if np.ndim(c.rho_k) > 0:
                    c.rho_k = update_rho(
                        c.rho_k, c.dual_res_k, c.primal_res_k, mu=mu_rho, tau=tau_rho
                    )
                else:
                    if individual_rho:
                        c.rho_k = update_rho(
                            c.rho_k,
                            np.linalg.norm(c.dual_res_k),
                            np.linalg.norm(c.primal_res_k),
                            mu=mu_rho,
                            tau=tau_rho,
                        )
                    else:
                        c.rho_k = update_rho(
                            c.rho_k, dual_err, primal_err, mu=mu_rho, tau=tau_rho
                        )

        info_here["cost"] = cost_original
        cost_history.append(cost_original)
        info_here.update(
            wrap_up(
                clique_list,
                cost_history,
                primal_err,
                dual_err,
                iter,
                early_stop,
                verbose,
            )
        )
        if info_here["stop"]:
            break

    X_k_list = [clique.X_new for clique in clique_list]
    return X_k_list, info_here


def solve_parallel(
    clique_list, X0=None, rho_start=RHO_START, early_stop=False, maxiter=MAXITER
):
    """Use ADMM to solve decomposed SDP, with simple parallelism."""

    def run_worker(clique, pipe):
        # ADMM loop.
        while True:
            g = pipe.recv()
            clique.g = g

            X, info = solve_inner_sdp(
                clique, rho=clique.rho_k, verbose=False, use_fusion=True
            )

            if X is not None:
                clique.X_new = X
                clique.counter += 1
                clique.status = 1
            else:
                clique.status = -1
                pass
            pipe.send(clique)

    initialize_z(clique_list, X0)

    # Setup the workers
    pipes = []
    procs = []
    for i, clique in enumerate(clique_list):
        clique.rho_k = rho_start
        clique.counter = 0
        clique.status = 0
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_worker, args=(clique, remote))]
        procs[-1].start()

    # Run ADMM
    rho_k = rho_start
    info_here = {"success": False, "msg": "did not converge", "stop": False}
    cost_history = []
    for iter in range(maxiter):
        # ADMM step 1: update X varaibles (in parallel)
        for k, pipe in enumerate(pipes):
            left = list(clique_list[k - 1].z_new) if k > 0 else []
            right = list(clique_list[k].z_new) if k < len(clique_list) - 1 else []
            pipe.send(cp.vstack([left, right]))
        clique_list = [pipe.recv() for pipe in pipes]

        # ADMM step 2: update Z variables
        update_z(clique_list)

        # ADMM step 3: update Lagrange multipliers
        primal_err, dual_err = update_sigmas(clique_list, rho_k)

        # Intermediate steps: update rho and check convergence
        rho_k = update_rho(rho_k, dual_err, primal_err)
        for clique in clique_list:
            clique.rho_k = rho_k

        cost_original = np.sum(
            [np.trace(clique.X_new @ clique.Q) for clique in clique_list]
        )
        cost_history.append(cost_original)
        info_here["cost"] = cost_original
        info_here.update(
            wrap_up(clique_list, cost_history, primal_err, dual_err, iter, early_stop)
        )
        if info_here["stop"]:
            break

    [p.terminate() for p in procs]
    return [clique.X_new for clique in clique_list], info_here
