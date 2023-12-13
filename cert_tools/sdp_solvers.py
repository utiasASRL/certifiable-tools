from copy import deepcopy
import sys

import casadi as cas
import cvxpy as cp
import mosek

import numpy as np
import scipy.sparse as sp


# Define global default values for MOSEK IP solver
sdp_opts_dflt = {}
sdp_opts_dflt["mosek_params"] = {
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
    "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
}
# sdp_opts_dflt["save_file"] = "solve_cvxpy_.ptf"


def adjust_Q(Q, offset=True, scale=True):
    from copy import deepcopy
    import scipy.sparse.linalg as spl

    # TODO(FD) choose if we are keeping this sanity check, might be useful
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
        Q_scale = spl.norm(Q_mat, "fro")
        # Q_scale = Q_mat.max()
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
    options=None,
    limit_constraints=False,
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
    S = cas.nlpsol("S", "ipopt", nlp)
    # Run Program
    sol_input = dict(lbg=g_rhs, ubg=g_rhs)
    if x_cand is not None:
        sol_input["x0"] = x_cand
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


def solve_sdp(
    Q,
    Constraints,
    adjust=False,
    use_primal=False,
    verbose=True,
    sdp_opts=sdp_opts_dflt,
    **kwargs,
):
    """Solve SDP using the MOSEK API.

    Args:
        Q (_type_): Cost Matrix
        Constraints (): List of tuples representing constraints. Each tuple, (A,b) is such that
                        tr(A @ X) == b
        adjust (bool, optional): Whether or not to rescale and shift Q for better conditioning.
        verbose (bool, optional): If true, prints output to screen. Defaults to True.

    Returns:
        _type_: _description_
    """
    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    sdp_opts_here = deepcopy(sdp_opts_dflt)
    sdp_opts_here.update(sdp_opts)
    sdp_opts_here["verbose"] = verbose

    if not use_primal:
        yvals = cp.Variable(len(Constraints))
        b = np.array([Ab[1] for Ab in Constraints])
        objective = cp.Maximize(-yvals @ b)
        H = cp.Variable(Q_here.shape, symmetric=True)
        H = cp.sum([Q_here] + [yvals[i] * Ab[0] for (i, Ab) in enumerate(Constraints)])
        constraints = [H >> 0]
        cprob = cp.Problem(objective, constraints)
        try:
            try:
                cprob.solve(
                    solver="MOSEK",
                    **sdp_opts_here,
                )
            except mosek.Error:
                print("Did not find MOSEK, using different solver.")
                cprob.solve(verbose=verbose, solver="CVXOPT")
        except Exception as e:
            cost = None
            X = None
            H = None
            yvals = None
            msg = "infeasible / unknown"
        else:
            if np.isfinite(cprob.value):
                cost = cprob.value
                X = constraints[0].dual_value
                H = H.value
                yvals = yvals.value
                msg = "converged"
            else:
                cost = None
                X = None
                H = None
                yvals = None
                msg = "unbounded"
    else:
        X = cp.Variable(Q.shape, symmetric=True)
        objective = cp.Minimize(cp.trace(Q_here @ X))
        constraints = [X >> 0]
        for A, b in Constraints:
            constraints += [cp.trace(A @ X) == b]

        cprob = cp.Problem(objective, constraints)
        try:
            sdp_opts_here["verbose"] = verbose
            try:
                cprob.solve(
                    solver="MOSEK",
                    **sdp_opts_here,
                )
            except mosek.Error:
                print("Did not find MOSEK, using different solver.")
                cprob.solve(verbose=verbose, solver="CVXOPT")
        except Exception as e:
            cost = None
            X = None
            H = None
            yvals = None
            msg = "infeasible / unknown"
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
    if verbose:
        print(msg)

    # reverse Q adjustment
    if cost:
        cost = cost * scale + offset
        yvals[0] -= offset
        yvals /= scale
    info = {"H": H, "yvals": yvals, "cost": cost, "msg": msg}
    return X, info


def solve_sdp_mosek(
    Q, Constraints, adjust=False, verbose=True, sdp_opts=sdp_opts_dflt, **kwargs
):
    """Solve SDP using the MOSEK API.

    Args:
        Q (_type_): Cost Matrix
        Constraints (): List of tuples representing constraints. Each tuple, (A,b) is such that
                        tr(A @ X) == b
        adjust (bool, optional): Whether or not to rescale and shift Q for better conditioning.
        verbose (bool, optional): If true, prints output to screen. Defaults to True.

    Returns:
        _type_: _description_
    """

    # Define a stream printer to grab output from MOSEK
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    with mosek.Task() as task:
        # Set log handler for debugging ootput
        if verbose:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        # Set options
        opts = sdp_opts["mosek_params"]

        task.putdouparam(
            mosek.dparam.intpnt_co_tol_pfeas, opts["MSK_DPAR_INTPNT_CO_TOL_PFEAS"]
        )
        task.putdouparam(
            mosek.dparam.intpnt_co_tol_rel_gap,
            opts["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"],
        )
        task.putdouparam(
            mosek.dparam.intpnt_co_tol_mu_red, opts["MSK_DPAR_INTPNT_CO_TOL_MU_RED"]
        )
        task.putdouparam(
            mosek.dparam.intpnt_co_tol_infeas, opts["MSK_DPAR_INTPNT_CO_TOL_INFEAS"]
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
        Q_l = sp.tril(Q_here, format="csr")
        rows, cols = Q_l.nonzero()
        vals = Q_l[rows, cols].tolist()[0]
        assert not np.any(np.isinf(vals)), ValueError("Cost matrix has inf vals")
        symq = task.appendsparsesymmat(dim, rows.astype(int), cols.astype(int), vals)
        task.putbarcj(0, [symq], [1.0])
        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)
        # Add constraints
        cnt = 0
        for A, b in Constraints:
            # Generate matrix
            A_l = sp.tril(A, format="csr")
            rows, cols = A_l.nonzero()
            vals = A_l[rows, cols].tolist()[0]
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
            barx = task.getbarxj(mosek.soltype.itr, 0)
            cost = task.getprimalobj(mosek.soltype.itr) * scale + offset
            yvals = np.array(task.getbarsj(mosek.soltype.itr, 0))
            X = np.zeros((dim, dim))
            cnt = 0
            for i in range(dim):
                for j in range(i, dim):
                    if j == 0:
                        X[i, i] = barx[cnt]
                    else:
                        X[j, i] = barx[cnt]
                        X[i, j] = barx[cnt]
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

        # H = Q_here - LHS.value
        # yvals = [x.value for x in y]

        # TODO(FD) can we read the dual variables from mosek solution?
        info = {"H": None, "yvals": yvals, "cost": cost, "msg": msg}
        # info = {"cost": cost, "msg": msg}
        return X, info


def solve_feasibility_sdp(
    Q,
    Constraints,
    x_cand,
    adjust=True,
    verbose=True,
    sdp_opts=sdp_opts_dflt,
    soft_epsilon=True,
    eps_tol=1e-8,
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

    sdp_opts_here = deepcopy(sdp_opts_dflt)
    sdp_opts_here.update(sdp_opts)
    sdp_opts_here["verbose"] = verbose

    H = cp.sum([Q] + [y[i] * Ai for (i, Ai) in enumerate(As)])
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
        try:
            cprob.solve(
                solver="MOSEK",
                **sdp_opts_here,
            )
        except mosek.Error:
            print("Did not find MOSEK, using different solver.")
            cprob.solve(verbose=verbose, solver="CVXOPT")
    except Exception as e:
        eps = None
        cost = None
        X = None
        H = None
        yvals = None
        msg = "infeasible / unknown"
    else:
        if np.isfinite(cprob.value):
            cost = cprob.value
            X = constraints[0].dual_value
            H = H.value
            yvals = [x.value for x in y]
            msg = "converged"
        else:
            eps = None
            cost = None
            X = None
            H = None
            yvals = None
            msg = "unbounded"
    if verbose:
        print(msg)

    # reverse Q adjustment
    if cost:
        cost = cost * scale + offset
        yvals[0] = yvals[0] * scale + offset
        H = Q_here + cp.sum([yvals[i] * Ai for (i, Ai) in enumerate(As)])
        eps = eps.value if soft_epsilon else eps_tol

    info = {"X": X, "yvals": yvals, "cost": cost, "msg": msg, "eps": eps}
    return H, info
