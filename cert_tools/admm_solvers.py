import sys
import time
from copy import deepcopy
from multiprocessing import Pipe, Process

import cvxpy as cp
import mosek.fusion as fu
import numpy as np

from cert_tools.admm_clique import ADMMClique, update_rho
from cert_tools.fusion_tools import mat_fusion, read_costs_from_mosek
from cert_tools.sdp_solvers import (
    adjust_Q,
    adjust_tol,
    adjust_tol_fusion,
    options_cvxpy,
    options_fusion,
)

EARLY_STOP = False
EARLY_STOP_MIN = 1e-3

# currently overwritten by lifters:
RHO_START = 1e5
MAXITER = 1000

# See eq. (3.13) in Boyd 2010 for explanations of these.
MU_RHO = 10.0  # how much dual and primal residual may get unbalanced.
TAU_RHO = 2.0  # how much to change rho in each iteration.
EPS_ABS = 0.0  # set to 0 to use relative only
EPS_REL = 1e-10
UPDATE_RHO = True  # if False, never update rho at all

# Stop ADMM if the last N_ADMM iterations don't have significant change in cost.
N_ADMM = 3

# Tolerance of inner SDP for ADMM
TOL_INNER = 1e-3
# Whether or not to adjust cost matrix.
ADJUST = False
# How to adjust the cost matrix (max for fro)
SCALE_METHOD = "max"

# Number of pipes to create for parallel ADMM implementation.
# Set to inf to create as many as cliques.
N_THREADS = 4  # np.inf


def initialize_admm(clique_list, X0=None):
    """Initialize g based on contents of X0 (initial feasible points)"""
    for k, clique in enumerate(clique_list):
        clique.counter = 0
        clique.status = 0
        if clique.G_dict is not None:
            if X0 is not None:
                # for debugging only
                clique.X = X0[clique.index]

                # just a sanity chak that X0 is actually valid.
                other = np.vstack(
                    [Gi @ X0[vi].reshape(-1, 1) for vi, Gi in clique.G_dict.items()]
                )
                clique.g = -clique.F @ X0[clique.index].flatten()
                np.testing.assert_allclose(other.flatten(), clique.g)
            else:
                clique.g = np.zeros(clique.F.shape[0])


def check_convergence(clique_list, primal_err, dual_err):
    """Check convergence using the criteria by [Boyd 2010]."""
    dim_primal = (len(clique_list) - 1) * clique_list[0].X_dim
    dim_dual = len(clique_list) * clique_list[0].X_dim
    eps_max = np.max(
        [
            np.max(np.hstack([clique.X_new.flatten(), clique.g]))
            for clique in clique_list[:-1]
        ]
    )
    eps_pri = np.sqrt(dim_primal) * EPS_ABS + EPS_REL * eps_max
    eps_dual = np.sqrt(dim_dual) * EPS_ABS + EPS_REL * np.linalg.norm(
        np.hstack([clique.sigmas for clique in clique_list])
    )
    return (primal_err < eps_pri) and (dual_err < eps_dual)


def update_g(clique_list):
    """Average the overlapping areas of X_new for consensus (stored in Z_new)."""
    for c in clique_list:
        c.g_prev = deepcopy(c.g)
        other_contributions = np.hstack(
            [Gi @ clique_list[vi].X_new.flatten() for vi, Gi in c.G_dict.items()]
        )
        c.g = 0.5 * (-c.F @ c.X_new.flatten() + other_contributions)
        assert c.g.shape == c.g_prev.shape


def update_sigmas(clique_list, rho_k):
    """Update sigmas (dual variables) and residual terms."""
    for clique in clique_list:
        assert isinstance(clique, ADMMClique)
        clique.primal_res_k = clique.F @ clique.X_new.flatten() + clique.g
        clique.dual_res_k = rho_k * (clique.g - clique.g_prev)
        clique.sigmas += rho_k * clique.primal_res_k


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
                print("iter     prim. error     dual error       cost")
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


def solve_inner_sdp_fusion(
    Q, Constraints, F, g, sigmas, rho, verbose=False, tol=1e-8, adjust=False
):
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
        if adjust:
            Q_here, scale, offset = adjust_Q(Q, scale_method=SCALE_METHOD)
        else:
            Q_here = Q
            offset = 0
            scale = 1.0
        if F is not None:
            assert g is not None
            F_fu = mat_fusion(F)
            err = fu.Expr.add(fu.Expr.mul(F_fu, fu.Expr.flatten(X)), g)

            # doesn't work unforuntately:
            # Expr.mul(0.5 * rho, Expr.sum(Expr.mulElm(err, err))),
            M.objective(
                fu.ObjectiveSense.Minimize,
                fu.Expr.add(
                    [
                        fu.Expr.dot(mat_fusion(Q_here), X),
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
            M.objective(fu.ObjectiveSense.Minimize, fu.Expr.dot(mat_fusion(Q_here), X))

        adjust_tol_fusion(options_fusion, tol)
        options_fusion["intpntCoTolRelGap"] = tol * 10
        for key, val in options_fusion.items():
            M.setSolverParam(key, val)
        if verbose:
            M.setLogHandler(sys.stdout)

        M.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
        M.solve()
        if M.getProblemStatus() in [
            fu.ProblemStatus.PrimalAndDualFeasible,
            fu.ProblemStatus.Unknown,
        ]:
            X = np.reshape(X.level(), Q.shape)
            cost = M.primalObjValue()
            cost = cost * scale + offset
            info = {
                "success": True,
                "cost": cost,
                "msg": f"solved with status {M.getProblemStatus()}",
            }
    return X, info


def solve_inner_sdp(
    clique: ADMMClique, rho=None, verbose=False, use_fusion=True, tol=1e-8, adjust=False
):
    """Solve the inner SDP of the ADMM algorithm, similar to [Dall'Anese 2013]

    Each clique j keeps track of g_j = [G_1 @ Z_1; ...; G_N @ Z_N] where the cliques Z_i, G_i
    designate clique that have overlap with j, such that F @ X_i = g_j.

    min <Q, X_j> + y'e(X_j) + rho/2*||e(X_j)||^2
    s.t. <Ai, X_j> = bi # primary
         X >= 0

    where e(X_j) = F @ vec(X_j) + g_j
    """
    if adjust:
        print("Warning: adjusting Q is currently not fully working for inner sdp")

    if use_fusion:
        # for debugging only
        err = clique.F @ clique.X.flatten() + clique.g
        # print(f"current error: {err}")

        return solve_inner_sdp_fusion(
            clique.Q,
            clique.Constraints,
            clique.F,
            clique.g,
            clique.sigmas,
            rho,
            verbose,
            tol=tol,
            adjust=adjust,
        )
    else:
        objective, scale, offset = clique.get_objective_cvxpy(
            clique.X_var, rho, adjust=adjust, scale_method=SCALE_METHOD
        )
        constraints = clique.get_constraints_cvxpy(clique.X_var)
        cprob = cp.Problem(objective, constraints)
        options_cvxpy["verbose"] = verbose
        adjust_tol(options_cvxpy, tol)
        options_cvxpy["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] = tol * 10
        try:
            cprob.solve(solver="MOSEK", accept_unknown=True, **options_cvxpy)
            cost = float(cprob.value) * scale + offset
            info = {
                "cost": cost,
                "success": clique.X_var.value is not None,
            }
        except Exception as e:
            info = {"cost": np.inf, "success": False}
        return clique.X_var.value, info


def solve_alternating(
    clique_list: list[ADMMClique],
    X0: list[np.ndarray] = None,
    sigmas: dict = None,
    rho_start: float = RHO_START,
    use_fusion: bool = True,
    early_stop: bool = EARLY_STOP,
    verbose: bool = False,
    adjust: bool = ADJUST,
    tol_inner=TOL_INNER,
    maxiter=MAXITER,
    mu_rho=MU_RHO,
    tau_rho=TAU_RHO,
):
    """Use ADMM to solve decomposed SDP, but without using parallelism."""
    if sigmas is not None:
        for k, sigma in sigmas.items():
            clique_list[k].sigmas = sigma
    rho_k = rho_start

    # rho_k = rho_start
    info_here = {"success": False, "msg": "did not converge.", "stop": False}

    cost_history = []

    initialize_admm(clique_list, X0)  # fill Z_new
    for iter in range(maxiter):
        cost_lagrangian = 0
        cost_original = 0

        # ADMM step 1: update X
        for k, clique in enumerate(clique_list):
            assert isinstance(clique, ADMMClique)  # for debugging only
            X, info = solve_inner_sdp(
                clique,
                rho_k,
                verbose=False,
                use_fusion=use_fusion,
                tol=tol_inner,
                adjust=adjust,
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
        update_g(clique_list)

        # ADMM step 3: update Lagrange multipliers
        update_sigmas(clique_list, rho_k)
        primal_err = np.linalg.norm(np.hstack([c.primal_res_k for c in clique_list]))
        dual_err = np.linalg.norm(np.hstack([c.dual_res_k for c in clique_list]))

        # update rho
        if UPDATE_RHO:
            rho_k = update_rho(rho_k, dual_err, primal_err, mu=mu_rho, tau=tau_rho)

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
    clique_list,
    X0=None,
    n_threads=N_THREADS,
    rho_start=RHO_START,
    early_stop=False,
    tol_inner=TOL_INNER,
    adjust: bool = ADJUST,
    maxiter=MAXITER,
    use_fusion=False,
    verbose=False,
    mu_rho=MU_RHO,
    tau_rho=TAU_RHO,
):
    """Use ADMM to solve decomposed SDP, with simple parallelism."""

    def run_worker(cliques_per_pipe, pipe):
        # signal thta the worker has been built (only for time measurements)
        assert pipe.recv() == 1
        pipe.send(1)

        while True:
            g_list = pipe.recv()
            for g, clique in zip(g_list, cliques_per_pipe):
                clique.g = g
                X, info = solve_inner_sdp(
                    clique,
                    clique.rho_k,
                    verbose=verbose,
                    use_fusion=use_fusion,
                    tol=tol_inner,
                    adjust=adjust,
                )

                if X is not None:
                    clique.X_new = X
                    clique.counter += 1
                    clique.status = 1
                else:
                    clique.status = -1
                    pass
            if cliques_per_pipe[0].index == 0:
                pass
            pipe.send([c.X_new for c in cliques_per_pipe])

    # Setup the workers
    n_pipes = min(n_threads, len(clique_list))
    boundaries = np.linspace(0, len(clique_list), n_pipes + 1)
    indices_per_pipe = {i: [] for i in range(n_pipes)}
    for i in range(len(clique_list)):
        k = np.where(boundaries <= i)[0][-1]
        indices_per_pipe[k].append(i)

    # Initialize z of all cliques
    initialize_admm(clique_list, X0)

    pipes = []
    procs = []
    lengths = []
    for k in range(n_pipes):
        cliques_per_pipe = [clique_list[i] for i in indices_per_pipe[k]]
        lengths.append(len(cliques_per_pipe))
        local, remote = Pipe()
        pipes.append(local)
        procs.append(
            Process(target=run_worker, args=(deepcopy(cliques_per_pipe), remote))
        )
        procs[-1].start()
    assert np.all(np.diff(lengths) < 2)

    # this is just to make sure we wait for all pipes to be set up, before
    # starting the timer.
    [pipe.send(1) for pipe in pipes]
    [pipe.recv() for pipe in pipes]

    t1 = time.time()

    # Run ADMM
    info_here = {"success": False, "msg": "did not converge", "stop": False}
    cost_history = []
    for iter in range(maxiter):
        # ADMM step 1: update X varaibles (in parallel)
        for k, pipe in enumerate(pipes):
            g_list = [clique_list[i].z_new for i in indices_per_pipe[k]]
            pipe.send(g_list)

        for k, pipe in enumerate(pipes):
            X_new_list = pipe.recv()
            for count, i in enumerate(indices_per_pipe[k]):
                if X_new_list[count] is not None:
                    clique_list[i].X_new = X_new_list[count]

        # ADMM step 2: update Z variables
        update_g(clique_list)

        # ADMM step 3: update Lagrange multipliers
        update_sigmas(clique_list)
        primal_err = np.linalg.norm(np.hstack([c.primal_res_k for c in clique_list]))
        dual_err = np.linalg.norm(np.hstack([c.dual_res_k for c in clique_list]))

        # Intermediate steps: update rho and check convergence
        if UPDATE_RHO:
            rho_new = update_rho(
                clique_list[0].rho_k, dual_err, primal_err, mu=mu_rho, tau=tau_rho
            )
            for c in clique_list:
                c.rho_k = rho_new
        cost_original = np.sum(
            [np.trace(clique.X_new @ clique.Q) for clique in clique_list]
        )
        cost_history.append(cost_original)
        info_here["cost"] = cost_original
        info_here.update(
            wrap_up(
                clique_list,
                cost_history,
                primal_err,
                dual_err,
                iter,
                early_stop,
                verbose=True,
            )
        )
        if info_here["stop"]:
            break

    info_here["time running"] = time.time() - t1
    info_here["cost history"] = cost_history

    [p.terminate() for p in procs]
    X_k_list = [clique.X_new for clique in clique_list]
    return X_k_list, info_here
