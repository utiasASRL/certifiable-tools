import sys
from copy import deepcopy

import casadi as cas
import cvxpy as cp
import mosek
import mosek.fusion as fu
import numpy as np
import scipy.sparse as sp
from cert_tools.fusion_tools import mat_fusion

# General tolerance parameter for SDPs (see "adjust_tol" function for its exact effect)
TOL = 1e-11

# for computing the lambda parameter, we are adding all possible constraints
# and therefore we might run into numerical problems. Setting below to a high value
# was found to lead to less cases where the solver terminates with "UNKNOWN" status.
# see https://docs.mosek.com/latest/pythonapi/parameters.html#doc-all-parameter-list
LAMBDA_REL_GAP = 0.1
LAMBDA_TOL = 1e-7  # looser tolerance for sparsity-promiting problem

# for the sparsity-promoting problem: |Hx| < EPSILON
# can set EPSILON to None, and we will minimize it.
EPSILON = None
# EPSILON = 1e-4

ADJUST = True  # adjust the matrix Q for better conditioning
PRIMAL = False  # governs how the problem is put into SDP solver

# normalize the Q matrix by either its Frobenius norm or the maximum value.
# this is done after extracting the biggest element (in upper-left corner, due
# to homogenization variable)
SCALE_METHOD = "max"  # max or fro

# Define global default values for MOSEK IP solver
options_cvxpy = {}
options_cvxpy["mosek_params"] = {
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": TOL,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": TOL,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": TOL,
    "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_PRIMAL",  # has no effect
}
# options_cvxpy["save_file"] = "solve_cvxpy.ptf"
options_fusion = {
    "intpntMaxIterations": 500,
    "intpntCoTolPfeas": TOL,
    "intpntCoTolDfeas": TOL,
    "intpntCoTolMuRed": TOL,
    "intpntSolveForm": "primal",  # has no effect
}


def adjust_tol(options, tol):
    options["mosek_params"].update(
        {
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": tol,
        }
    )


def adjust_tol_fusion(options, tol):
    options.update(
        {
            "intpntCoTolPfeas": tol,
            "intpntCoTolDfeas": tol,
            "intpntCoTolMuRed": tol,
        }
    )


def adjust_Q(Q, scale=True, offset=True, scale_method=SCALE_METHOD):
    """
    :returns: Q scaled, scale, offset.
    """
    ii, jj = (Q == Q.max()).nonzero()
    if (ii[0], jj[0]) != (0, 0) or (len(ii) > 1):
        print(
            "Warning: largest element of Q is not unique or not in top-left. Check ordering?"
        )

    Q_mat = deepcopy(Q)
    if offset:
        Q_offset = Q_mat[0, 0]
    else:
        Q_offset = 0
    Q_mat[0, 0] -= Q_offset

    if scale:
        if scale_method == "fro":
            try:
                Q_scale = sp.linalg.norm(Q_mat, "fro")
            except TypeError:
                Q_scale = np.linalg.norm(Q_mat, ord="fro")
        elif scale_method == "max":
            Q_scale = abs(Q_mat).max()
        else:
            raise ValueError("Unknown cost scaling method")
    else:
        Q_scale = 1.0
    Q_mat /= Q_scale
    return Q_mat, Q_scale, Q_offset


def get_subgradient(Q, A_list, a):
    from cert_tools.eig_tools import get_min_eigpairs

    H_curr = Q + np.sum([ai * Ai for ai, Ai in zip(a, A_list)])

    eig_vals, eig_vecs = get_min_eigpairs(H_curr)
    U = 1 / Q.shape[0] * np.eye(Q.shape[0])
    return eig_vecs @ U @ eig_vecs.T


def solve_low_rank_sdp(
    Q,
    Constraints,
    rank=1,
    x_cand=None,
    adjust=(1, 0),
    options={},
    limit_constraints=False,
    verbose=False,
):
    """Use the factorization proposed by Burer and Monteiro to solve a
    fixed rank SDP.
    """
    # Get problem dimensions
    n = Q.shape[0]
    # Define variable
    Y = cas.SX.sym("Y", n, rank)
    # Define cost
    f = cas.trace(Y.T @ Q @ Y)
    # Define constraints
    g_lhs = []
    g_rhs = []
    for A, b in Constraints:
        g_lhs += [cas.trace(Y.T @ A @ Y)]
        g_rhs += [b]
    # Limit the number of constraints used to the degrees of freedom
    if limit_constraints and len(g_lhs) > rank * n:
        g_lhs = g_lhs[: rank * n]
        g_rhs = g_rhs[: rank * n]
    # Concatenate
    g_lhs = cas.vertcat(*g_lhs)
    g_rhs = cas.vertcat(*g_rhs)
    # Define Low Rank NLP
    nlp = {"x": Y.reshape((-1, 1)), "f": f, "g": g_lhs}
    options["ipopt.print_level"] = int(verbose)
    options["print_time"] = int(verbose)
    S = cas.nlpsol("S", "ipopt", nlp, options)
    # Run Program
    sol_input = dict(lbg=g_rhs, ubg=g_rhs)
    if x_cand is not None:
        sol_input["x0"] = x_cand.reshape((-1, 1))
    r = S(**sol_input)
    Y_opt = r["x"]
    # Reshape and generate SDP solution
    Y_opt = np.array(Y_opt).reshape((n, rank), order="F")
    X_opt = Y_opt @ Y_opt.T
    # Get cost
    scale, offset = adjust
    cost = np.trace(Q @ X_opt) * scale + offset
    # Construct certificate
    mults = np.array(r["lam_g"])
    H = Q
    for i, (A, b) in enumerate(Constraints):
        if limit_constraints and i >= n * rank:
            break
        H = H + A * mults[i, 0]

    # Return
    info = {"X": X_opt, "H": H, "cost": cost}
    return Y_opt, info


def solve_sdp_mosek(
    Q,
    Constraints,
    adjust=ADJUST,
    primal=PRIMAL,
    tol=TOL,
    verbose=True,
    options=options_cvxpy,
):
    """Solve SDP using the MOSEK API.

    Args:
        Q: Cost matrix
        Constraints: List of tuples representing constraints. Each tuple, (A,b) is such that
                        tr(A @ X) == b
        adjust (bool, optional): Whether or not to rescale and shift Q for better conditioning.
        verbose (bool, optional): If true, prints output to screen. Defaults to True.

    Returns:
        (X, info, cost_out): solution matrix, info dict and output cost.
    """
    # WARNING: THIS SEEMS THE WRONG WAY AROUND, BUT THIS IS IN ACCORDANCE
    # WITH WHAT CVXPY CALLS PRIMAL VS. DUAL!
    if primal:
        print("Warning: cannot use primal formulation for mosek API (yet).")

    # Define a stream printer to grab output from MOSEK
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    if tol:
        adjust_tol(options, tol)

    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    with mosek.Task() as task:
        # Set log handler for debugging ootput
        if verbose:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        # Set options
        opts = options["mosek_params"]

        task.putdouparam(
            mosek.dparam.intpnt_co_tol_pfeas, opts["MSK_DPAR_INTPNT_CO_TOL_PFEAS"]
        )
        task.putdouparam(
            mosek.dparam.intpnt_co_tol_mu_red, opts["MSK_DPAR_INTPNT_CO_TOL_MU_RED"]
        )
        task.putdouparam(
            mosek.dparam.intpnt_co_tol_dfeas, opts["MSK_DPAR_INTPNT_CO_TOL_DFEAS"]
        )
        # problem params
        dim = Q_here.shape[0]
        numcon = len(Constraints)
        # append vars,constr
        task.appendbarvars([dim])
        task.appendcons(numcon)
        # bound keys
        bkc = mosek.boundkey.fx
        # Cost matrix
        Q_l = sp.tril(Q_here)
        rows, cols = Q_l.coords
        vals = Q_l.data
        assert not np.any(np.isinf(vals)), ValueError("Cost matrix has inf vals")
        symq = task.appendsparsesymmat(dim, rows.astype(int), cols.astype(int), vals)
        task.putbarcj(0, [symq], [1.0])
        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)
        # Add constraints
        cnt = 0
        for A, b in Constraints:
            # Generate matrix
            A_l = sp.tril(A)
            rows, cols = A_l.coords
            vals = A_l.data
            syma = task.appendsparsesymmat(dim, rows, cols, vals)
            # Add constraint matrix
            task.putbaraij(cnt, 0, [syma], [1.0])
            # Set bound (equality)
            task.putconbound(cnt, bkc, b, b)
            cnt += 1
        # Store problem
        task.writedata("solve_mosek.ptf")
        # Solve the problem and print summary
        task.optimize()
        task.solutionsummary(mosek.streamtype.msg)
        # Get status information about the solution
        prosta = task.getprosta(mosek.soltype.itr)
        solsta = task.getsolsta(mosek.soltype.itr)
        if solsta == mosek.solsta.optimal:
            msg = "Optimal"
            # Primal variable
            barx = task.getbarxj(mosek.soltype.itr, 0)
            # Problem Cost
            cost = task.getprimalobj(mosek.soltype.itr) * scale + offset
            # Lagrange Multipliers
            yvals = task.gety(mosek.soltype.itr)
            # Dual SDP
            bars = task.getbarsj(mosek.soltype.itr, 0)
            # Convert back
            X = np.zeros((dim, dim))
            S = np.zeros((dim, dim))
            cnt = 0
            for i in range(dim):
                for j in range(i, dim):
                    if j == 0:
                        X[i, i] = barx[cnt]
                        S[i, i] = bars[cnt]
                    else:
                        X[j, i] = barx[cnt]
                        X[i, j] = barx[cnt]
                        S[j, i] = bars[cnt]
                        S[i, j] = bars[cnt]
                    cnt += 1
        elif (
            solsta == mosek.solsta.dual_infeas_cer
            or solsta == mosek.solsta.prim_infeas_cer
        ):
            msg = "Primal or dual infeasibility certificate found.\n"
            X = np.nan
            cost = np.nan
        elif solsta == mosek.solsta.unknown:
            msg = "Unknown solution status"
            X = np.nan
            cost = np.nan
        else:
            msg = f"Other solution status: {solsta}"
            X = np.nan
            cost = np.nan
        # Return Additional information
        info = {"H": S, "yvals": yvals, "cost": cost, "msg": msg}
        # info = {"cost": cost, "msg": msg}
        return X, info


def solve_sdp_fusion(
    Q,
    Constraints,
    B_list=[],
    adjust=ADJUST,
    primal=PRIMAL,
    tol=TOL,
    verbose=False,
    options=options_fusion,
):
    """Run Mosek's Fusion API to solve a semidefinite program.

    See solve_sdp_mosek for argument description.
    """

    if len(B_list):
        raise ValueError("cannot deal with B_list yet.")

    if tol:
        adjust_tol_fusion(options, tol)

    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    # WARNING: THIS SEEMS THE WRONG WAY AROUND, BUT THIS IS IN ACCORDANCE
    # WITH WHAT CVXPY CALLS PRIMAL VS. DUAL!
    if not primal:
        with fu.Model("dual") as M:
            # creates (N x X_dim x X_dim) variable
            X = M.variable("X", fu.Domain.inPSDCone(Q.shape[0]))

            # standard equality constraints
            for A, b in Constraints:
                M.constraint(fu.Expr.dot(mat_fusion(A), X), fu.Domain.equalsTo(b))

            M.objective(fu.ObjectiveSense.Minimize, fu.Expr.dot(mat_fusion(Q_here), X))

            if verbose:
                M.setLogHandler(sys.stdout)

            for key, val in options.items():
                M.setSolverParam(key, val)

            M.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
            M.solve()

            if M.getProblemStatus() in [
                fu.ProblemStatus.PrimalAndDualFeasible,
                fu.ProblemStatus.Unknown,
            ]:
                cost = M.primalObjValue() * scale + offset
                H = np.reshape(X.dual(), Q.shape)
                X = np.reshape(X.level(), Q.shape)
                msg = f"success with status {M.getProblemStatus()}"
                success = True
            else:
                cost = None
                H = None
                X = None
                msg = f"solver failed with status {M.getProblemStatus()}"
                success = False
            info = {"success": success, "cost": cost, "msg": msg, "H": H}
            return X, info
    else:
        # TODO(FD) below is extremely slow and runs out of memory for 200 x 200 matrices.
        with fu.Model("primal") as M:
            m = len(Constraints)
            b = fu.Matrix.dense(np.array([-b for A, b in Constraints])[None, :])
            y = M.variable("y", [m, 1])

            # standard equality constraints
            H = fu.Expr.add(
                mat_fusion(Q_here),
                fu.Expr.add(
                    [
                        fu.Expr.mul(mat_fusion(Constraints[i][0]), y.index([i, 0]))
                        for i in range(m)
                    ]
                ),
            )
            con = M.constraint(H, fu.Domain.inPSDCone(Q.shape[0]))
            M.objective(
                fu.ObjectiveSense.Maximize,
                fu.Expr.sum(fu.Expr.mul(b, y)),
            )

            if verbose:
                M.setLogHandler(sys.stdout)

            for key, val in options.items():
                M.setSolverParam(key, val)

            M.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
            M.solve()

            if M.getProblemStatus() in [
                fu.ProblemStatus.PrimalAndDualFeasible,
                fu.ProblemStatus.Unknown,
            ]:
                cost = M.primalObjValue() * scale + offset
                X = np.reshape(con.dual(), Q.shape)
                if X[0, 0] < 1:
                    X = -X
                msg = f"success with status {M.getProblemStatus()}"
                success = True
            else:
                cost = None
                X = None
                msg = "solver failed"
                success = False
            info = {"success": success, "cost": cost, "msg": msg}
    return X, info


def solve_sdp_cvxpy(
    Q,
    Constraints,
    B_list=[],
    adjust=ADJUST,
    primal=PRIMAL,
    tol=TOL,
    verbose=False,
    options=options_cvxpy,
):
    """Run CVXPY with MOSEK to solve a semidefinite program.

    See solve_sdp_mosek for argument description.
    """

    if tol:
        adjust_tol(options, tol)
    options["verbose"] = verbose

    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    As, b = zip(*Constraints)

    if primal:
        """
        min < Q, X >
        s.t.  trace(Ai @ X) == bi, for all i.
        """
        X = cp.Variable(Q.shape, symmetric=True)
        constraints = [X >> 0]
        constraints += [cp.trace(A @ X) == b for A, b in Constraints]
        constraints += [cp.trace(B @ X) <= 0 for B in B_list]
        cprob = cp.Problem(cp.Minimize(cp.trace(Q_here @ X)), constraints)
        try:
            cprob.solve(
                solver="MOSEK",
                **options,
            )
        except cp.SolverError as e:
            cost = None
            X = None
            H = None
            yvals = None
            msg = f"infeasible / unknown: {e}"
        else:
            if np.isfinite(cprob.value):
                cost = cprob.value
                X = X.value
                H = constraints[0].dual_value
                yvals = [c.dual_value for c in constraints[1:]]
                msg = "converged"
            else:
                cost = None
                X = None
                H = None
                yvals = None
                msg = "unbounded"
    else:  # Dual
        """
        max < y, b >
        s.t. sum(Ai * yi for all i) << Q
        """
        m = len(Constraints)
        y = cp.Variable(shape=(m,))

        k = len(B_list)
        if k > 0:
            u = cp.Variable(shape=(k,))

        b = np.concatenate([np.atleast_1d(bi) for bi in b])
        objective = cp.Maximize(b @ y)

        # We want the lagrangian to be H := Q - sum l_i * A_i + sum u_i * B_i.
        # With this choice, l_0 will be negative
        LHS = cp.sum(
            [y[i] * Ai for (i, Ai) in enumerate(As)]
            + [-u[i] * Bi for (i, Bi) in enumerate(B_list)]
        )
        # this does not include symmetry of Q!!
        constraints = [LHS << Q_here]
        if k > 0:
            constraints.append(u >= 0)

        cprob = cp.Problem(objective, constraints)
        try:
            cprob.solve(
                solver="MOSEK",
                **options,
            )
        except cp.SolverError as e:
            cost = None
            X = None
            H = None
            yvals = None
            msg = f"infeasible / unknown: {e}"
        else:
            if np.isfinite(cprob.value):
                cost = cprob.value
                X = constraints[0].dual_value
                H = Q_here - LHS.value
                yvals = [x.value for x in y]

                # sanity check for inequality constraints.
                # we want them to be inactive!!!
                if len(B_list):
                    mu = np.array([ui.value for ui in u])
                    i_nnz = np.where(mu > 1e-10)[0]
                    if len(i_nnz):
                        for i in i_nnz:
                            print(
                                f"Warning: is constraint {i} active? (mu={mu[i]:.4e}):"
                            )
                            print(np.trace(B_list[i] @ X))
                msg = "converged"
            else:
                cost = None
                X = None
                H = None
                yvals = None
                msg = "unbounded"

    # reverse Q adjustment
    if cost:
        cost = cost * scale + offset

        H = Q_here - cp.sum(
            [yvals[i] * Ai for (i, Ai) in enumerate(As)]
            + [-u[i] * Bi for (i, Bi) in enumerate(B_list)]
        )
        yvals[0] = yvals[0] * scale + offset
        # H *= scale
        # H[0, 0] += offset

    info = {"H": H, "yvals": yvals, "cost": cost, "msg": msg}
    return X, info


def solve_feasibility_sdp(
    Q,
    Constraints,
    x_cand,
    adjust=ADJUST,
    tol=None,
    soft_epsilon=True,
    eps_tol=1e-8,
    verbose=False,
    options=options_cvxpy,
):
    """Solve feasibility SDP using the MOSEK API.

    Args:
        Q (_type_): Cost Matrix
        Constraints (): List of tuples representing constraints. Each tuple, (A,b) is such that
                        tr(A @ X) == b.
        x_cand (): Solution candidate.
        adjust (tuple, optional): Adjustment tuple: (scale,offset) for final cost.
        verbose (bool, optional): If true, prints output to screen. Defaults to True.

    Returns:
        _type_: _description_
    """
    m = len(Constraints)
    y = cp.Variable(shape=(m,))

    As, b = zip(*Constraints)
    b = np.concatenate([np.atleast_1d(bi) for bi in b])

    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    if tol:
        adjust_tol(options, tol)
    options["verbose"] = verbose

    H = cp.sum([Q_here] + [y[i] * Ai for (i, Ai) in enumerate(As)])
    constraints = [H >> 0]
    if soft_epsilon:
        eps = cp.Variable()
        constraints += [H @ x_cand <= eps]
        constraints += [H @ x_cand >= -eps]
        objective = cp.Minimize(eps)
    else:
        eps = cp.Variable()
        constraints += [H @ x_cand <= eps_tol]
        constraints += [H @ x_cand >= -eps_tol]
        objective = cp.Minimize(1.0)

    cprob = cp.Problem(objective, constraints)
    try:
        cprob.solve(solver="MOSEK", accept_unknown=True, **options)
    except Exception as e:
        eps = None
        cost = None
        X = None
        H = None
        yvals = None
        msg = "infeasible / unknown"
    else:
        if np.isfinite(cprob.value):
            eps = eps.value if soft_epsilon else eps_tol
            cost = cprob.value
            X = constraints[0].dual_value
            H = H.value
            yvals = [x.value for x in y]
            msg = f"converged: {cprob.status}"
        else:
            eps = None
            cost = None
            X = None
            H = None
            yvals = None
            msg = f"unbounded: {cprob.status}"
    if verbose:
        print(msg)

    # reverse Q adjustment
    if cost:
        cost = cost * scale + offset
        yvals[0] = yvals[0] * scale + offset
        H = Q_here + cp.sum([yvals[i] * Ai for (i, Ai) in enumerate(As)])

    info = {"X": X, "yvals": yvals, "cost": cost, "msg": msg, "eps": eps}
    return H, info
