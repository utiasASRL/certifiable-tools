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
# Plotting
import matplotlib.pyplot as plt
# Data storage
import pandas as pd

# Default options for penalty optimization
opts_dflt = dict(tol_eig = 1e-8,
                    tol_pen = 1e-8,
                    max_iter = 1000,
                    rho = 100,
                    btrk_c = 0.5,
                    btrk_rho = 0.5,
                    grad_sqr_tol=1e-14,
                    tol_cost = 1e-14
                    )

# Default options for cutting plane method
opts_cut_dflt = dict(tol_eig = 1e-6,            # Eigenvalue tolerance
                     max_iter=1000,             # Maximum iterations
                     min_eig_ub = 1.,          # Upper bound for cutting plane
                     lambda_level = 0.9,        # level multiplier (for level method)
                     level_method_bound = 1e5,   # above this level, default to vanilla cut plane
                     tol_null = 1e-5,           # null space tolerance for first order KKT constraints
                     use_null = True,           # if true, reparameterize problem using null space
                     cut_buffer = 30,           # number of cuts stored at once, FIFO
                     )            

# PROJECTED GRADIENT METHOD

def null_project(A, x):
    """Iterative null-space projection (leverages matrix vector products)"""
    # Solve for the part of x that is perpendicular to the null space
    x_perp = sp.linalg.lsqr(A, A @ x)[0]
    x_perp = np.expand_dims(x_perp, axis=1)
    # return x with perp component removed.
    return x - x_perp

def solve_eopt_project(Q, Constraints, x_cand, opts=opts_dflt, verbose=True):
    """Solve the certificate/eigenvalue optimization problem using a projection to
    force the step onto the constraint manifold. """
    # Convert candidate solution to sparse
    if not sp.issparse(x_cand):
        x_cand = sp.csc_array(x_cand)
    # Get affine constraint from constraints and candidate solution
    A_bar = sp.hstack([A @ x_cand for A,b in Constraints])
    b_bar = -Q @ x_cand
    A_list = [A for A,b in Constraints]
    # Initialize
    cost_prev = -np.inf
    grad_sqr = 0.
    alpha = 1
    status = 'run'
    header_printed=False
    n_iter = 0
    x = sp.linalg.lsqr(A_bar, b_bar.toarray(),atol=opts['tol_pen'])[0]
    x = np.expand_dims(x,axis=1)
    while status == 'run':
        # Construct Certificate matrix
        H = Q + np.sum([ai[0] * Ai for ai, (Ai,b) in zip(x.tolist(), Constraints)])
        # Compute eigenvalue function subgrad
        sgrad_eig, min_eig, _, t = get_grad_info(H, A_list, method="shifted-lanczos", k=6)
        # compute overall cost
        cost = min_eig
        # Update step with backtracking
        if n_iter == 0 or cost >= cost_prev + opts['btrk_c']*grad_sqr*alpha:
            # Sufficient increase, accept new iterate
            x_prev = x.copy()
            alpha_prev = alpha
            alpha = 1.
            cost_prev = cost
            n_iter += 1
            # Compute gradient at new iterate
            grad  = null_project(A_bar, sgrad_eig)
            grad_sqr = (grad.T @ grad)[0,0]
            # print output
            if verbose:
                if header_printed is False:
                    print(' N   |   grad^2   |   cost     |   alpha    | mult |')
                    header_printed=True
                print(f" {n_iter:3d} | {grad_sqr:5.4e} | {cost_prev:5.4e} | {alpha_prev:5.4e} | {t:4d}")
                # print(f"grad:  {grad.T}")
            # Check termination criteria
            if min_eig >= -opts['tol_eig']:
                status = 'opt_found'
            elif grad_sqr < opts['grad_sqr_tol']:
                status = 'grad_zero'
            elif n_iter > opts['max_iter']:
                status = 'max_iter'
            if not status == 'run':
                break
        else: 
            # increase not sufficient, backtrack.
            alpha = alpha * opts['btrk_rho']
        # Generate a new test point 
        x = x_prev + alpha * grad
    # output info
    info = dict(n_iter=n_iter,
                status=status,
                multipliers=x,
                min_eig=min_eig)
    return H, info

# CUTTING PLANE METHOD
def get_grad_info(H,
                A_vec,
                U=None,
                tau=1e-8,
                get_hessian=False,
                damp_hessian=True,
                **kwargs):
    eig_vals, eig_vecs = get_min_eigpairs(H, **kwargs)
    # get minimum eigenvalue     
    min_eig = np.min(eig_vals)
    # split eigenvector sets based on closeness to min (multiplicity could be > 1)
    ind_1 = np.abs(eig_vals - min_eig) < tau
    Q_1 = eig_vecs[:,ind_1]
    # Size of matrix
    n = H.shape[0]
    # Multiplicity
    t = Q_1.shape[1]
    # Scale variable
    if U is None:
        U = 1 / t * np.eye(t)
    # Compute gradient
    subgrad = A_vec.T @ (Q_1 @ U @ Q_1.T).reshape(-1,1,order='F')
    # Compute Hessian 
    if t == 1 and get_hessian:
        # Get other eigvectors and eigenvalues
        ind_s = np.abs(eig_vals - min_eig) >= tau
        Q_s = eig_vecs[:,ind_s] 
        eig_inv_diffs = 1/(min_eig - eig_vals[ind_s])
        Lambda_s = np.diag(eig_inv_diffs)
        # Construct Hessian
        Q_bar = A_vec.T @ np.kron(np.eye(n),Q_1) @ Q_s
        hessian = Q_bar @ Lambda_s @ Q_bar.T * 2        
        # Compute damping term for conditioning of QP
        damp = np.min(eig_inv_diffs)
    else:
        hessian = None
        damp = None
    grad_info = dict(subgrad=subgrad,
                     hessian=hessian,
                     min_eig=min_eig,
                     min_vec=Q_1,
                     multplct=t,
                     damp=damp)
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
    for A,b in Constraints:
        # LHS matrix of affine constraint
        A_bar += [A @ x_cand]
        # Vectorized list of constraints: TODO Change this to the half-vec format
        A_vec += [A.reshape((-1,1),order='F')]
    A_bar = np.hstack(A_bar)
    A_vec = sp.hstack(A_vec)
    # RHS of affine constraints
    b_bar = -C @ x_cand
    
    # Perform QR decomposition to characterize and work with null space
    Q,R,P = scipy.linalg.qr(A_bar, pivoting=True)
    r = np.abs(np.diag(R))
    rank = np.sum(r > opts['tol_null'])
    Q1 = Q[:,:rank]
    R1 = R[:rank,:]   
    # Inverse permutation
    Pinv = np.zeros(len(P),int)
    for k,p in enumerate(P):
        Pinv[p] = k
    if use_null: # eliminate affine constraints based on null space
        # Based on Section 5.5.5 "Basic Solutions via QR with Column Pivoting" from Golub and Van Loan.
        R11, R12 = R1[:rank, :rank], R1[:rank, rank:]
        N = np.vstack([sla.solve_triangular(R11, R12), -np.eye(R12.shape[1])])
        basis = np.zeros(N.T.shape)
        basis[:, P] = N.T
        # New constraint set
        A_vec_null = A_vec @ basis.T
    else:
        A_vec_null = None
    # Truncate eigenvalues of A_bar to make nullspace more defined (required due to opt tolerances)   
    # Define equality constraints. Column of zeros added for epigraph form variable
    # TODO Consider adding this column later when actually running the LP
    A_eq = np.hstack([np.zeros((R1.shape[0], 1)),R1[:,Pinv]])
    b_eq = Q1.T @ b_bar
    # Output 
    return A_vec, A_eq, b_eq, A_vec_null, basis.T
    
# @profile
def solve_eopt_cuts(C, Constraints, x_cand, opts=opts_cut_dflt, verbose=True,**kwargs):
    """Solve the certificate/eigenvalue optimization problem using a cutting plane algorithm.
    Current algorithm uses the level method with the target level at a tolerance below zero"""
    # Preprocess constraints
    use_null = opts['use_null']
    constr_info = preprocess_constraints(C,
                                        Constraints,
                                        x_cand,
                                        use_null=use_null,
                                        opts=opts)
    A_vec, A_eq, b_eq, A_vec_null,basis = constr_info
    # Get orthogonal vector to x_cand
    x_rand = np.random.rand(*x_cand.shape)-1
    x_orth = x_rand - x_cand.T@x_rand / (x_cand.T@x_cand)
    # INITIALIZE
     # Initialize multiplier variables
    if use_null:
        x_bar = la.lstsq(A_eq[:,1:], b_eq)[0]
        x = 2*np.ones((A_vec_null.shape[1],1))
    else:
        x = la.lstsq(A_eq[:,1:], b_eq)[0]
    # Init cutting plane constraints 
    #    t <= level
    # <==>   [ 1 0 ]@[t;x] <= level
    one = np.array([[1.]])
    A_cut = None
    b_cut = None
    # Intialize status vars for optimization
    status = 'RUNNING'
    header_printed=False
    n_iter = 0
    t_max = np.inf
    t_min = -np.inf
    iter_info = []
    # LOOP
    while status == "RUNNING":
        # SOLVE CUT PLANE PROGRAM WITH MAX LEVEL
        # Cost: maximize t
        c = np.hstack([-one, np.zeros((1,len(x)))])
        # Set bounds on t level only
        bounds = [(None,opts['min_eig_ub'])] + [(None,None)]*len(x)
        # Solve opt problem
        if use_null:
            A_eq_lp = None
            b_eq_lp = None
        else:
            A_eq_lp = A_eq
            b_eq_lp = b_eq.squeeze()
        res = linprog(c=c.squeeze(),
                      A_eq=A_eq_lp,
                      b_eq=b_eq_lp,
                      A_ub=A_cut,
                      b_ub=b_cut,
                      method='highs-ds',
                      bounds=bounds,
                      )
        if res.success:
            t_lp, x_lp = res.x[0],res.x[1:]
            x_lp = np.expand_dims(x_lp,axis=1)
        else:
            raise ValueError("Linear subproblem failed.")
        # COMPUTE LEVEL PROJECTION
        level = t_min + opts['lambda_level']*(t_lp - t_min)
        # Condition on level required for now for solving projection
        if n_iter > 0 and np.abs(level) <= opts['level_method_bound'] and opts['lambda_level'] < 1:
            t_x = level_project(x_prox=x,
                                level=level,
                                A_eq=A_eq_lp,
                                b_eq=b_eq_lp,
                                A_ub=A_cut,
                                b_ub=b_cut,)
            t_qp, x_new = t_x[0,:],t_x[1:,:]
        else:
            x_new = x_lp
        # CUT PLANE UPDATE
        # Store previous gradient information
        if n_iter > 0:
            grad_info_prev = grad_info.copy()
        # Number of eigenvalues to compute
        k = 10
        if use_null:
            # Construct Current Certificate matrix
            H = get_cert_mat(C, A_vec, x_bar,A_vec_null, x_new)
            # current gradient and minimum eig
            grad_info = get_grad_info(H=H, A_vec=A_vec_null, k=k,method="direct",v0=x_orth)
        else:
            # Construct Current Certificate matrix
            H = get_cert_mat(C, A_vec, x_new)
            # current gradient and minimum eig
            grad_info = get_grad_info(H=H, A_vec=A_vec, k=k,method="direct",v0=x_orth)
        # Check minimum eigenvalue
        if grad_info['min_eig'] > t_min:
            t_min = grad_info['min_eig']
        # Add Cuts
        # NOTE: t <= grad @ (x-x_bar) + f(x_bar) <==> 
        # [1 -grad] @ [t; x] <= f(x_bar) -grad @ x_bar  
        a_cut = np.hstack([one, -grad_info['subgrad'].T])
        b_val = grad_info['min_eig'] - grad_info['subgrad'].T @ x_new
        if A_cut is None:
            A_cut = a_cut
            b_cut = np.array(b_val)
        else:
            A_cut = np.vstack([A_cut, a_cut])
            b_cut = np.hstack([b_cut, b_val])
            if A_cut.shape[0] > opts['cut_buffer']:
                A_cut = A_cut[1:,:]
                b_cut = b_cut[:,1:]
        # STATUS UPDATE
        # update model upper bound
        if t_lp <= t_max: 
            t_max = t_lp
        # define gap
        gap = (t_max-t_min)
        # termination criteria
        if t_min >= -opts['tol_eig']:
            status = "POS_LB"
        elif t_max < -2*opts['tol_eig']:
            status = "NEG_UB"
        elif n_iter >= opts['max_iter']:
            status = "MAX_ITER"
        # Gradient delta
        if n_iter > 0:
            delta_grad = grad_info['subgrad'] - grad_info_prev['subgrad']
        else:
            delta_grad = np.zeros(grad_info['subgrad'].shape)
        # Update vars
        n_iter += 1
        delta_x = x_new - x
        delta_norm = la.norm(delta_x)
        #plot_along_grad(C, A_vec, x_bar,A_vec_null, x_new,grad_info['subgrad'],1)
        x = x_new
        # Curvature
        curv = (delta_grad.T @ delta_x)[0,0] / delta_norm
        # PRINT
        if verbose:
            if n_iter % 10 == 1:
                header_printed=False
            if header_printed is False:
                print(' N   | delta_nrm |  eig val  |   t_max   |   t_min     |   gap    |   curv   | mult. |')
                header_printed=True
            print(f" {n_iter:3d} | {delta_norm:5.4e} | {grad_info['min_eig']:5.4e} | {t_max:5.4e} | {t_min:5.4e} | {gap:5.4e} | {curv:5.4e} | {grad_info['multplct']:4d} ")
            # Store data
            info=dict(n_iter=n_iter,
                    delta_norm=delta_norm,
                    x = x,
                    min_eig_curr=grad_info['min_eig'],
                    t_max=t_max,
                    t_min=t_min,
                    gap=gap,
                    mult=grad_info['multplct'],
                    curv = curv,
                    )
            iter_info+=[info]
    # Final set of multipliers
    if use_null:
        mults = x_bar + basis @ x   
    else:
        mults = x 
    
    # Set outputs
    output = dict(H=H,
                  status=status,
                  mults=mults,
                  gap=gap,
                  t_min=t_min,
                  t_max=t_max,
                  iter_info=pd.DataFrame(iter_info),
                  cuts = (A_cut, b_cut),
                  A_vec = A_vec,
                  A_vec_null = A_vec_null,
                  x=x
                )
    
    return output

def level_project(x_prox, level, A_eq, b_eq, A_ub, b_ub):
    """Solve level set projection problem: return the closest point to x_prox 
    (the prox-center) such that the acheived model value is above the provided
    level"""
    # check dims
    if not b_eq is None and len(b_eq.shape) == 1:
        b_eq = np.expand_dims(b_eq,axis=1)
    # Cost:
    q = np.vstack([0., -x_prox])
    P = np.eye(len(x_prox)+1)
    P[0,0] = 0.
    # Modify constraints to add level lower bound
    # -t <= level
    a_ub = np.vstack([-1., np.zeros(x_prox.shape)]).T
    if A_ub is None:
        G = a_ub
        h = -level
    else:
        G = np.vstack([a_ub, A_ub])
        h = np.hstack([np.array([[-level]]), b_ub]).T
    # define vars
    x = cp.Variable((len(x_prox)+1,1),'x')
    # define constraints
    constrs = [G @ x <= h]
    if not A_eq is None:
        constrs += [A_eq @ x == b_eq]
    # Solve Quadratic Program:
    prob = cp.Problem(cp.Minimize(cp.norm2(x[1:,:]-x_prox)), constrs)
    prob.solve(solver='MOSEK', verbose=False)
    # Get solution
    return x.value
    
def get_cert_mat(C, A_vec, mults,A_vec_null=None, mults_null=None, sparsify=True):
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
    H += (A_vec @ mults).reshape(H.shape,order='F')
    if not A_vec_null is None:
        # Add null space constraints
        H += (A_vec_null @ mults_null).reshape(H.shape,order='F')
    return H

def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    """ Solves the quadratic program:
    min  1/2 x^T P x +  q^T
    s.t. G @ x <= h
         A @ x = b"""
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    return res[0]
        
def cvxpy_qp(P, q, G, h, A=None, b=None, verbose=True):
    x = cp.Variable((P.shape[0],1),'x')
    prob = cp.Problem(cp.Minimize(0.5*cp.quad_form(x, P) + q.T @ x),
                    [G @ x <= h,
                    A @ x == b])
    options = dict(feastol=1e-5,reltol=1e-5,abstol=1e-5)
    options = dict()
    prob.solve(verbose=verbose, solver='CVXOPT', **options)
    return x.value

def plot_along_grad(C, A_vec, mults, A_vec_null, mults_null,step, alpha_max):
    alphas = np.linspace(0,alpha_max,100)
    min_eigs = np.zeros(alphas.shape)
    for i in range(len(alphas)):
        step_alpha = mults_null + alphas[i]*step
        # Apply step
        H_alpha = get_cert_mat(C, A_vec, mults, A_vec_null, step_alpha)
        # Check new minimum eigenvalue
        grad_info = get_grad_info(H_alpha,A_vec,k=10,method="direct")
        min_eigs[i] = grad_info['min_eig']
        
    # Plot min eig
    plt.figure()
    plt.plot(alphas, min_eigs, color='r')
    plt.show()
        
    
    
    return f