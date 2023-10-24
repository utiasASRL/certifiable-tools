# Optimization
import mosek
import cvxpy as cp
from scipy.optimize import linprog
import scipy
import gurobipy as gp
import quadprog

# Maths
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.sparse as sp
import sparseqr as sqr
from cert_tools.eig_tools import get_min_eigpairs
from cert_tools.linalg_tools import get_nullspace

# Plotting
import matplotlib.pyplot as plt

# Data storage
import pandas as pd

# Default options for penalty optimization
opts_dflt = dict(
    tol_eig=1e-8,
    tol_pen=1e-8,
    max_iter=1000,
    rho=100,
    btrk_c=0.5,
    btrk_rho=0.5,
    grad_sqr_tol=1e-14,
    tol_cost=1e-14,
)

# Default options for cutting plane method
opts_cut_dflt = dict(
    tol_eig=1e-6,  # Eigenvalue tolerance
    max_iter=1000,  # Maximum iterations
    min_eig_ub=1.0,  # Upper bound for cutting plane
    lambda_level=0.9,  # level multiplier (for level method)
    level_method_bound=1e5,  # above this level, default to vanilla cut plane
    tol_null=1e-5,  # null space tolerance for first order KKT constraints
    use_null=True,  # if true, reparameterize problem using null space
    cut_buffer=30,  # number of cuts stored at once, FIFO
    use_hessian=False,  # Flag to select whether to use the Hessian
)


class CutPlaneModel:
    """This class stores all of the information of the cutting plane model. This
    includes all of the cutting planes as well as any equality constraints.
    This class also stores methods for finding the optimum of the model.
    Model in epigraph form is as follows:
    max t
    s.t. t <= values[j] + gradients[j].T @ delta
                    + gradients[j].T @(x_current - eval_pts[j]), forall j
    """

    def __init__(m, n_vars, A_eq=None, b_eq=None, opts=opts_cut_dflt):
        """Initialize cut plane model"""
        # Store shape
        m.n_vars = n_vars
        # Add a single cutting plane
        m.gradients = []
        m.values = []
        m.eval_pts = []
        m.n_cuts = 0
        # Equality constraints
        m.A_eq = A_eq
        m.b_eq = b_eq
        # Cutting plane buffer limit
        m.opts = opts

    def add_cut(m, grad_info: dict, eval_pt):
        """Add a cutting plane to the model

        Args:
            m (CutPlaneModel)
            grad_info (dict): gradient information dictionary
            eval_pt (): point where the gradient was evaluated
        """
        # Add data
        m.gradients += [grad_info["subgrad"]]
        m.values += [grad_info["min_eig"]]
        m.eval_pts += [eval_pt]
        # Update number of cuts
        m.n_cuts += 1
        # Remove cuts if exceeding the max number allowed
        if m.n_cuts > m.opts["cut_buffer"]:
            m.rm_cut(ind=0)

    def rm_cut(m, ind):
        """Remove cut planes at index"""
        # Remove planes
        m.gradients.pop(ind)
        m.values.pop(ind)
        m.eval_pts.pop(ind)
        # Decrement cut counter
        m.n_cuts -= 1

    def evaluate(m, x):
        values = [
            (m.values[i] + m.gradients[i].T @ (x - m.eval_pts[i]))[0, 0]
            for i in range(m.n_cuts)
        ]
        return min(values + [m.opts["min_eig_ub"]])

    def solve_lp_linprog(m, use_null):
        # Cost: maximize t
        one = np.array([[1.0]])
        c = -np.hstack([one, np.zeros((1, m.n_vars))])
        # Set bounds on t level only
        bounds = [(None, m.opts["min_eig_ub"])] + [(None, None)] * m.n_vars
        # Solve opt problem
        if use_null:
            A_eq_lp = None
            b_eq_lp = None
        else:
            A_eq_lp = np.hstack([np.zeros((m.A_eq.shape[0], 1)), m.A_eq])
            b_eq_lp = m.b_eq.squeeze()
        # NOTE: t <= grad @ (x-x_j) + f(x_j) <==>
        # [1 -grad] @ [t; x] <= f(x_j) -grad @ x_j
        A_cut = np.hstack((np.ones((m.n_cuts, 1)), -np.hstack(m.gradients).T))
        b_cut = np.array(
            [m.values[i] - m.gradients[i].T @ m.eval_pts[i] for i in range(m.n_cuts)]
        )
        # Run Linprog
        res = linprog(
            c=c.squeeze(),
            A_eq=A_eq_lp,
            b_eq=b_eq_lp,
            A_ub=A_cut,
            b_ub=b_cut,
            method="highs-ds",
            bounds=bounds,
        )
        if res.success:
            t_lp, x_lp = res.x[0], res.x[1:]
            x_lp = np.expand_dims(x_lp, axis=1)
        else:
            raise ValueError("Linear subproblem failed.")
        return t_lp, x_lp

    def solve_level_project(m, x_prox, level):
        """Solve level set projection problem: return the closest point to x_prox
        (the prox-center) such that the acheived model value is above the provided
        level"""
        # VARIABLES
        delta = cp.Variable((m.n_vars, 1), "delta")
        t = cp.Variable(1, "t")
        # CONSTRAINTS
        # cut plane model constraints
        # NOTE: these constraints could be cached
        constraints = [
            t <= m.values[i] + m.gradients[i].T @ (delta + x_prox - m.eval_pts[i])
            for i in range(m.n_cuts)
        ]
        # model value larger than specified level
        constraints += [t >= level]
        # add equality constraints
        if not m.A_eq is None:
            if len(m.b_eq.shape) == 1:
                b_eq = m.b_eq[:, None]
            else:
                b_eq = m.b_eq
            constraints += [m.A_eq @ (delta + x_prox) == b_eq]
        # Solve Quadratic Program:
        prob = cp.Problem(cp.Minimize(cp.norm2(delta)), constraints)
        prob.solve(solver="MOSEK", verbose=False)
        # Get solution
        return t.value, delta.value + x_prox


def get_grad_info(
    H, A_vec, U=None, tau=1e-8, get_hessian=False, damp_hessian=True, **kwargs
):
    eig_vals, eig_vecs = get_min_eigpairs(H, **kwargs)
    # get minimum eigenvalue
    min_eig = np.min(eig_vals)
    # split eigenvector sets based on closeness to min (multiplicity could be > 1)
    ind_1 = np.abs(eig_vals - min_eig) < tau
    Q_1 = eig_vecs[:, ind_1]
    # Size of matrix
    n = H.shape[0]
    # Multiplicity
    t = Q_1.shape[1]
    # Scale variable
    if U is None:
        U = 1 / t * np.eye(t)
    # Compute gradient
    subgrad = A_vec.T @ (Q_1 @ U @ Q_1.T).reshape(-1, 1, order="F")
    # Compute Hessian
    if t == 1 and get_hessian:
        # Get other eigvectors and eigenvalues
        ind_s = np.abs(eig_vals - min_eig) >= tau
        Q_s = eig_vecs[:, ind_s]
        eig_inv_diffs = 1 / (min_eig - eig_vals[ind_s])
        Lambda_s = np.diag(eig_inv_diffs)
        # Construct Hessian
        Q_bar = A_vec.T @ np.kron(np.eye(n), Q_1) @ Q_s
        hessian = Q_bar @ Lambda_s @ Q_bar.T * 2
        # Compute damping term for conditioning of QP
        damp = np.min(eig_inv_diffs)
    else:
        hessian = None
        damp = None
    grad_info = dict(
        subgrad=np.array(subgrad),
        hessian=hessian,
        min_eig=min_eig,
        min_vec=Q_1,
        multplct=t,
        damp=damp,
    )
    return grad_info


def preprocess_constraints(C, Constraints, x_cand, use_null=False, opts=opts_cut_dflt):
    """Pre-processing steps for certificate optimization.
    Uses cost, constraints, and candidate solution to build the affine, first-order
    conditions for the SDP.
    If option use_null set to True, the constraints are reparameterized using
    a sparse null-space basis for the original optimality conditions

    Args:
        C (_type_): _description_
        Constraints (_type_): _description_
        x_cand (_type_): _description_
        opts (_type_, optional): _description_. Defaults to opts_cut_dflt.
    """
    # Loop through constraints
    A_bar = []
    A_vec = []
    for A, b in Constraints:
        # LHS matrix of affine constraint
        A_bar += [A @ x_cand]
        # Vectorized list of constraints: TODO Change this to the half-vec format
        A_vec += [A.reshape((-1, 1), order="F")]
    A_bar = np.hstack(A_bar)
    A_vec = sp.hstack(A_vec)
    # RHS of affine constraints
    b_bar = -C @ x_cand

    # Perform QR decomposition to characterize and work with null space
    basis, info = get_nullspace(A_bar, method="qrp", tolerance=opts["tol_null"])

    if use_null:  # eliminate affine constraints based on null space
        # New constraint set
        A_vec_null = A_vec @ basis.T
    else:
        A_vec_null = None
    # Truncate eigenvalues of A_bar to make nullspace more defined (required due to opt tolerances)
    # Define equality constraints. Column of zeros added for epigraph form variable
    # TODO Consider adding this column later when actually running the LP
    A_eq = info["LHS"]
    b_eq = info["Q1"].T @ b_bar
    # Output
    return A_vec, A_eq, b_eq, A_vec_null, basis.T


def solve_eopt_cuts(Q, Constraints, x_cand, opts=opts_cut_dflt, verbose=True, **kwargs):
    """Solve the certificate/eigenvalue optimization problem using a cutting plane algorithm.
    Current algorithm uses the level method with the target level at a tolerance below zero
    """
    # Preprocess constraints
    use_null = opts["use_null"]
    constr_info = preprocess_constraints(
        Q, Constraints, x_cand, use_null=use_null, opts=opts
    )
    A_vec, A_eq, b_eq, A_vec_null, basis = constr_info
    # Get orthogonal vector to x_cand
    x_rand = np.random.rand(*x_cand.shape) - 1
    x_orth = x_rand - x_cand.T @ x_rand / (x_cand.T @ x_cand)
    # INITIALIZE
    # Initialize multiplier variables
    if use_null:
        x_bar = la.lstsq(A_eq, b_eq, rcond=None)[0]
        x = np.ones((A_vec_null.shape[1], 1))
    else:
        x = kwargs.get("x_init", la.lstsq(A_eq, b_eq)[0])

    # Init cutting plane model
    if not use_null:
        m = CutPlaneModel(x.shape[0], A_eq=A_eq, b_eq=b_eq)
    else:
        m = CutPlaneModel(x.shape[0])
    # Intialize status vars for optimization
    status = "RUNNING"
    header_printed = False
    n_iter = 0
    t_max = np.inf
    t_min = -np.inf
    iter_info = []
    # LOOP
    while status == "RUNNING":
        # SOLVE CUT PLANE PROGRAM
        if n_iter > 0:
            if opts["use_hessian"]:
                # Solve QP
                pass
            else:
                # Solve LP
                t_lp, x_lp = m.solve_lp_linprog(use_null)
                # Compute level projection
                level = t_min + opts["lambda_level"] * (t_lp - t_min)
                # Condition on level required for now for solving projection
                if (
                    np.abs(level) <= opts["level_method_bound"]
                    and opts["lambda_level"] < 1
                ):
                    t_qp, x_new = m.solve_level_project(x_prox=x, level=level)
                else:
                    x_new = x_lp
            # Store previous gradient information
            grad_info_prev = grad_info.copy()
        else:
            # Initialization step
            x_new = x.copy()
            t_lp = m.evaluate(x_new)
        # CUT PLANE UPDATE
        # Number of eigenvalues to compute
        k = 10
        if use_null:
            # Construct Current Certificate matrix
            H = get_cert_mat(Q, A_vec, x_bar, A_vec_null, x_new)
            # current gradient and minimum eig
            grad_info = get_grad_info(
                H=H, A_vec=A_vec_null, k=k, method="direct", v0=x_orth
            )
        else:
            # Construct Current Certificate matrix
            H = get_cert_mat(Q, A_vec, x_new)
            # current gradient and minimum eig
            grad_info = get_grad_info(H=H, A_vec=A_vec, k=k, method="direct", v0=x_orth)
        # Check minimum eigenvalue
        if grad_info["min_eig"] > t_min:
            t_min = grad_info["min_eig"]
        # Add Cuts
        m.add_cut(grad_info, x_new)

        # STATUS UPDATE
        # update model upper bound
        if t_lp <= t_max:
            t_max = t_lp
        # define gap
        gap = t_max - t_min
        # termination criteria
        if t_min >= -opts["tol_eig"]:
            status = "POS_LB"
        elif t_max < -2 * opts["tol_eig"]:
            status = "NEG_UB"
        elif n_iter >= opts["max_iter"]:
            status = "MAX_ITER"
        # Gradient delta
        if n_iter > 0:
            delta_grad = grad_info["subgrad"] - grad_info_prev["subgrad"]
        else:
            delta_grad = np.zeros(grad_info["subgrad"].shape)
        # Update vars
        n_iter += 1
        delta_x = x_new - x
        delta_norm = la.norm(delta_x)
        plot_along_grad(Q, A_vec, x_bar, A_vec_null, x_new, grad_info["subgrad"], 1)
        x = x_new
        # Curvature
        if delta_norm > 0:
            curv = (delta_grad.T @ delta_x)[0, 0] / delta_norm
        else:
            curv = 0.0
        # PRINT
        if verbose:
            if n_iter % 10 == 1:
                header_printed = False
            if header_printed is False:
                print(
                    " N   | delta_nrm |  eig val  |   t_max   |   t_min     |   gap    |   curv   | mult. |"
                )
                header_printed = True
            print(
                f" {n_iter:3d} | {delta_norm:5.4e} | {grad_info['min_eig']:5.4e} | {t_max:5.4e} | {t_min:5.4e} | {gap:5.4e} | {curv:5.4e} | {grad_info['multplct']:4d} "
            )
            # Store data
            info = dict(
                n_iter=n_iter,
                delta_norm=delta_norm,
                x=x,
                min_eig_curr=grad_info["min_eig"],
                t_max=t_max,
                t_min=t_min,
                gap=gap,
                mult=grad_info["multplct"],
                curv=curv,
            )
            iter_info += [info]
    # Final set of multipliers
    if use_null:
        mults = x_bar + basis @ x
    else:
        mults = x

    # Set outputs
    output = dict(
        H=H,
        status=status,
        mults=mults,
        gap=gap,
        t_min=t_min,
        t_max=t_max,
        iter_info=pd.DataFrame(iter_info),
        model=m,
        A_vec=A_vec,
        A_vec_null=A_vec_null,
        x=x,
    )

    return output


def get_cert_mat(C, A_vec, mults, A_vec_null=None, mults_null=None, sparsify=True):
    """Generate certificate matrix from cost, constraints and multipliers
    C is the cost matrix amd A_vec is expected to be a vectorized version
    of the constraint matrices"""
    if sp.issparse(C) and not sparsify:
        H = C.todense()
    elif not sp.issparse(C) and sparsify:
        H = sp.csc_array(C)
    else:
        H = C.copy()
    # Loop through A matrices
    if sp.issparse(A_vec) and not sparsify:
        A_vec = A_vec.todense()
    elif not sp.issparse(A_vec) and sparsify:
        A_vec = sp.csc_array(A_vec)
    # Add in standard constraints
    H += (A_vec @ mults).reshape(H.shape, order="F")
    if not A_vec_null is None:
        # Add null space constraints
        H += (A_vec_null @ mults_null).reshape(H.shape, order="F")
    return H


def plot_along_grad(C, A_vec, mults, A_vec_null, mults_null, step, alpha_max):
    alphas = np.linspace(0, alpha_max, 100)
    min_eigs = np.zeros(alphas.shape)
    for i in range(len(alphas)):
        step_alpha = mults_null + alphas[i] * step
        # Apply step
        H_alpha = get_cert_mat(C, A_vec, mults, A_vec_null, step_alpha)
        # Check new minimum eigenvalue
        grad_info = get_grad_info(H_alpha, A_vec, k=10, method="direct")
        min_eigs[i] = grad_info["min_eig"]

    # Plot min eig
    plt.figure()
    plt.plot(alphas, min_eigs, color="r")
    plt.show()
