# Optimization
import mosek
import cvxpy as cp
import casadi as cas
# Maths
import numpy as np
import scipy.sparse as sp
# 
import sys, os


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
sdp_opts_dflt["save_file"] = "solve_cvxpy_.ptf"


def solve_low_rank_sdp(Q,
                       Constraints,
                       rank=1,
                       Y_0=None,
                       adjust=(1,0),
                       options=None,
                       limit_constraints=False):
    """Use the factorization proposed by Burer and Monteiro to solve a
    fixed rank SDP.
    """
    # Get problem dimensions
    n = Q.shape[0]
    # Define variable
    Y = cas.SX.sym('Y',n,rank)
    # Define cost
    f = cas.trace(Y.T @ Q @ Y)
    # Define constraints
    g_lhs = []
    g_rhs = []
    for A,b in Constraints:
        g_lhs += [cas.trace(Y.T @ A @ Y)]
        g_rhs += [b]
    # Limit the number of constraints used to the degrees of freedom
    if limit_constraints and len(g_lhs) > rank*n:
        g_lhs = g_lhs[:rank*n]
        g_rhs = g_rhs[:rank*n]
    # Concatenate
    g_lhs = cas.vertcat(*g_lhs)
    g_rhs = cas.vertcat(*g_rhs)
    # Define Low Rank NLP
    nlp = {'x' : Y.reshape((-1,1)),
           'f' : f,
           'g' : g_lhs}
    S = cas.nlpsol('S', 'ipopt', nlp)
    # Run Program
    sol_input = dict(lbg=g_rhs, ubg=g_rhs)
    if not Y_0 is None:
        sol_input['x0'] = Y_0
    r = S(**sol_input)
    Y_opt = r['x']
    # Reshape and generate SDP solution
    Y_opt = np.array(Y_opt).reshape((n,rank),order='F')
    X_opt = Y_opt @ Y_opt.T
    # Get cost
    scale, offset = adjust
    cost=np.trace(Q @ X_opt) * scale + offset
    # Construct certificate
    mults = np.array(r['lam_g'])
    H = Q
    for i,(A,b) in enumerate(Constraints):
        if limit_constraints and i >= n*rank:
            break
        H = H + A * mults[i,0]
        
    
    # Return
    return Y_opt, X_opt, H, cost

def solve_sdp_mosek(Q,
                    Constraints,
                    adjust=(1.0, 0.0),
                    verbose=True,
                    sdp_opts=sdp_opts_dflt):
    """Solve SDP using the MOSEK API.

    Args:
        Q (_type_): Cost Matrix
        Constraints (): List of tuples representing constraints. Each tuple, (A,b) is such that
                        tr(A @ X) == b
        adjust (tuple, optional): Adjustment tuple: (scale,offset) for final cost.
        verbose (bool, optional): If true, prints output to screen. Defaults to True.

    Returns:
        _type_: _description_
    """

    # Define a stream printer to grab output from MOSEK
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

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
        scale, offset = adjust
        dim = Q.shape[0]
        numcon = len(Constraints)
        # append vars,constr
        task.appendbarvars([dim])
        task.appendcons(numcon)
        # bound keys
        bkc = mosek.boundkey.fx
        # Cost matrix
        Q_l = sp.tril(Q, format="csr")
        rows, cols = Q_l.nonzero()
        vals = Q_l[rows,cols].tolist()[0]
        assert not np.any(np.isinf(vals)), ValueError("Cost matrix has inf vals")
        symq = task.appendsparsesymmat(dim,
                                        rows.astype(int),
                                        cols.astype(int),
                                        vals)
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
            barx = task.getbarxj(mosek.soltype.itr, 0)
            cost = task.getprimalobj(mosek.soltype.itr) * scale + offset
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
            print("Primal or dual infeasibility certificate found.\n")
            X = np.nan
            cost = np.nan
        elif solsta == mosek.solsta.unknown:
            print("Unknown solution status")
            X = np.nan
            cost = np.nan
        else:
            print("Other solution status")
            X = np.nan
            cost = np.nan
        return X, cost
