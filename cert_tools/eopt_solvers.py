# Optimization
import mosek
import cvxpy as cp
from scipy.optimize import linprog

# Maths
import numpy as np
import scipy.sparse as sp
import sparseqr as sqr
from cert_tools.eig_tools import get_min_eigpairs
# Plotting
import matplotlib.pyplot as plt

# Default options for penalty optimization
opts_dflt = dict(tol_eig = 1e-8,
                    tol_pen = 1e-8,
                    max_iter = 100,
                    rho = 100,
                    btrk_c = 0.5,
                    btrk_rho = 0.5,
                    grad_sqr_tol=1e-14,
                    tol_cost = 1e-14
                    )

# Default options for cutting plane method
opts_cut_dflt = dict(tol_eig = 1e-8,
                     max_iter=100,
                     method = 'level',
                     level=1e1)


def solve_eopt_penalty(Q, Constraints, x_cand, opts=opts_dflt, verbose=True):
    """Solve the certificate/eigenvalue optimization problem using a penalty term
    to enforce the affine constraints (i.e. first order opt conditions). """
    # Convert candidate solution to sparse
    if not sp.issparse(x_cand):
        x_cand = sp.csc_array(x_cand)
    # Get affine constraint from constraints and candidate solution
    A_bar = sp.hstack([A @ x_cand for A,b in Constraints])
    b_bar = Q @ x_cand
    A_list = [A for A,b in Constraints]
    # Initialize
    cost_prev = -np.inf
    grad_sqr = 0.
    alpha = 1
    status = 'run'
    header_printed=False
    n_iter = 0
    x = sp.linalg.lsqr(A_bar, -b_bar.toarray(),atol=opts['tol_pen'])[0]
    x = np.expand_dims(x,axis=1)
    while status == 'run':
        # Construct Certificate matrix
        H = Q + np.sum([ai[0] * Ai for ai, (Ai,b) in zip(x.tolist(), Constraints)])
        # Compute eigenvalue function subgrad
        sgrad_eig, min_eig = get_grad_info(H, A_list, method="direct", k=6)
        # Compute penalty value and gradient
        err_penalty = A_bar @ x + b_bar
        cost_penalty = -(err_penalty.T @ err_penalty) * opts['rho']
        grad_penalty = -A_bar.T @ err_penalty * opts['rho']
        # compute overall cost
        cost = min_eig + cost_penalty[0,0]
        # Check for sufficient increase
        if n_iter == 0 or cost >= cost_prev + (opts['btrk_c']*grad_sqr*alpha):
            # Update the current optimal values
            x_prev = x.copy()
            alpha_prev = alpha
            alpha = 1.
            cost_prev = cost
            grad = sgrad_eig + grad_penalty
            grad_sqr = (grad.T @ grad)[0,0]
            n_iter += 1
            # print output
            if verbose:
                if header_printed is False:
                    print(' N   |   grad^2   |   cost    |   alpha   ')
                    header_printed=True
                print(f" {n_iter:3d} | {grad_sqr:5.4e} | {cost_prev:5.4e} | {alpha_prev:5.4e}")
                print(f"grad:  {grad.T}")
            # Check termination criteria
            if min_eig >= -opts['tol_eig'] and cost_penalty / opts['rho'] >= -opts['tol_pen']:
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
                cost_penalty=cost_penalty,
                min_eig=min_eig)
    return H, info

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

# SEQUENTIAL GRADIENT DESCENT METHOD

def get_grad_info(H,
                    A_list,
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
    # Multiplicity
    t = Q_1.shape[1]
    # Scale variable
    if U is None:
        U = 1 / t * np.eye(t)
    # Compute gradient
    subgrad = np.vstack([np.trace(U @ (Q_1.T @ Ai @ Q_1)) for Ai in A_list])
    # Compute Hessian 
    m = len(A_list)
    if t == 1 and get_hessian:
        # Get other eigvectors and eigenvalues
        ind_s = np.abs(eig_vals - min_eig) >= tau
        Q_s = eig_vecs[:,ind_s] 
        eig_inv_diffs = 1/(min_eig - eig_vals[ind_s])
        Lambda_s = np.diag(eig_inv_diffs)
        # Construct Hessian
        Q_bar = np.vstack([Q_1.T @ A_k @ Q_s for A_k in A_list])
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

def solve_step_qp(grad_info, A, b, use_LM=False):
    """Solve Quadratic Program leveraging matrix vector products. 
    Expected that hessian is low rank and negative semidefinite.
    Based on eq 18.9 of Nocedal and Wright"""
    grad = grad_info['subgrad']
    hess0 = grad_info['hessian']
    # Sparsity checks
    if not sp.issparse(hess0):
        hess0 = sp.csc_array(hess0)
    if sp.issparse(b):
        b = b.todense()
    # Levinberg-Marquardt
    if use_LM:
        hess = hess0 + sp.eye(hess0.shape[0])*grad_info['damp']
    else:
        hess = hess0
    # Construct coefficient matrix
    B1 = sp.hstack([hess, A.T])
    B2 = sp.hstack([ A, sp.csc_array((A.shape[0],A.shape[0])) ])
    B = sp.vstack([B1,B2])
    # Construct rhs
    c = np.array(np.vstack([-grad, b]))
    # Compute solution
    solution = sp.linalg.lsqr(B, c,atol=1e-9, btol=1e-9)
    x_opt = np.expand_dims(solution[0], axis=1)
    step, mult = x_opt[:hess.shape[0],[0]], x_opt[hess.shape[0]:,[0]]
    # Compute the expected delta
    cost_delta = step.T @ hess @ step/2 + grad.T @ step
    return step, cost_delta[0,0]              

def solve_eopt_sqp(Q, Constraints, x_cand, opts=opts_dflt, verbose=True):
    """Solve the certificate/eigenvalue optimization problem by constructing an SQP
    Assumes that the multiplicity of the minimum eigenvalue is always 1."""
    # Convert candidate solution to sparse
    if not sp.issparse(x_cand):
        x_cand = sp.csc_array(x_cand)
    # Get affine constraint from constraints and candidate solution
    A_bar = sp.hstack([A @ x_cand for A,b in Constraints])
    b_bar = -Q @ x_cand
    A_list = [A for A,b in Constraints]
    # Initialize
    cost_prev = -np.inf
    cost_delta=0
    alpha = 1
    status = 'RUNNING'
    header_printed=False
    n_iter = 0
    # x = sp.linalg.lsqr(A_bar, b_bar.toarray(),atol=opts['tol_pen'])[0]
    # x = np.expand_dims(x,axis=1)
    x = np.zeros((A_bar.shape[1],1))
    # Opt Loop:
    while status == 'RUNNING':
        # Construct Certificate matrix
        H = Q + np.sum([ai[0] * Ai for ai, (Ai,b) in zip(x.tolist(), Constraints)])
        # Compute eigenvalue function subgrad
        k = 100
        grad_info = get_grad_info(H,
                                  A_list,
                                  get_hessian=True,
                                  method="direct",
                                  k=k)
        # Overall cost
        cost = grad_info['min_eig']
        # Update step with backtracking 
        # ****** BACKTRACKING OFF FOR NOW
        if True: # n_iter == 0 or cost >= cost_prev + cost_delta*alpha:
            # Sufficient increase, accept new iterate
            x_prev = x.copy()
            alpha_prev = alpha
            alpha = 1.
            cost_prev = cost
            # Run solve quadratic program to get update step,
            # Solve first step with actual 
            if n_iter > 0:
                b=b_bar*0
                step, cost_delta = solve_step_qp(grad_info=grad_info,A=A_bar,b=b)
            else:
                b=b_bar
                grad_info['hessian']=sp.csc_array(grad_info['hessian'].shape)
                step, cost_delta = solve_step_qp(grad_info=grad_info,A=A_bar,b=b,use_LM=False)
            # print output
            if verbose:
                if header_printed is False:
                    print(' N   |   cost inc   |   cost     |   alpha    | mult |')
                    header_printed=True
                print(f" {n_iter:3d} | {cost_delta:5.4e} | {cost_prev:5.4e} | {alpha_prev:5.4e} | {grad_info['multplct']:4d}")
            # Check termination criteria
            if False:#n_iter>0:
                if cost >= -opts['tol_eig']:
                    status = 'POS_MIN_EIG'
                elif cost_delta < opts['tol_cost']:
                    status = 'COST_DELTA_NOT_POS'
                elif n_iter > opts['max_iter']:
                    status = 'MAX_ITER'
                if not status == 'RUNNING':
                    break
            n_iter += 1
        else: 
            # increase not sufficient, backtrack.
            alpha = alpha * opts['btrk_rho']
        # Generate a new test point 
        x = x_prev + alpha * step
    
    # Output info
    res_constr = np.linalg.norm((H@x_cand).todense())
    if verbose:
        print("Summary:")
        print(f"  Final cost (min eig):  {cost:5.4e} ")
        print(f"  Constraint residual:  {res_constr:5.4e} ")
    info = dict(n_iter=n_iter,
                status=status,
                multipliers=x,
                grad_info=grad_info,
                min_eig=cost,
                res_constr=res_constr)
    return H, info

def solve_eopt_cuts(Q, Constraints, x_cand, opts=opts_cut_dflt, verbose=True,**kwargs):
    """Solve the certificate/eigenvalue optimization problem using a cutting plane algorithm.
    Current algorithm uses the level method with the target level at a tolerance below zero"""
    # Get affine constraint from constraints and candidate solution
    A_bar = np.hstack([A @ x_cand for A,b in Constraints])
    b_bar = -Q @ x_cand
    A_list = [A for A,b in Constraints]
    # Truncate eigenvalues of A_bar to make nullspace more defined
    Q,R = np.linalg.qr(A_bar)
    b_eq = Q.T@b_bar
    tol = 1e-5
    for i,r in enumerate(np.diag(R)):
        if np.abs(r) < tol:
            break
    # Define affine constraints in epigraph form
    A_eq = np.hstack([np.zeros((i, 1)), R[:i,:]])
    b_eq = Q.T@b_bar
    b_eq = b_eq[:i,:]

    # INITIALIZE
    status = 'RUNNING'
    header_printed=False
    n_iter = 0
    t_max = np.inf
    t_min = -np.inf
    # Initialize multiplier variables
    # x = np.linalg.lstsq(R[:i,:], b_eq)[0]
    x = np.zeros((len(Constraints),1))
    # Init cutting plane constraints 
    #    t <= level
    # <==>   [ 1 0 ]@[t;x] <= level
    one = np.array([[1.]])
    A_cut = None
    b_cut = None
    # LOOP
    while status == "RUNNING":
        # CUT PLANE UPDATE
        # Construct Current Certificate matrix
        H = Q.copy()
        for i,(A, b) in enumerate(Constraints):
            H += x[i] * A
        # current gradient and minimum eig
        grad_info = get_grad_info(H=H, A_list=A_list, k=4,method="direct")
        if grad_info['min_eig'] > t_min:
            t_min = grad_info['min_eig']
        # NOTE: t <= grad @ (x-x_bar) + f(x_bar) <==> 
        # [1 -grad] @ [t; x] <= f(x_bar) -grad @ x_bar  
        a_cut = np.hstack([one, -grad_info['subgrad'].T])
        b_val = grad_info['min_eig'] - grad_info['subgrad'].T @ x
        if A_cut is None:
            A_cut = a_cut
            b_cut = b_val
        else:
            A_cut = np.vstack([A_cut, a_cut])
            b_cut = np.vstack([b_cut, b_val])
        # SOLVE CUT PLANE PROGRAM WITH MAX LEVEL
        # Warm start        
        x0 = np.vstack([t_min-1e-4, x])
        # Cost: maximize t
        c = np.hstack([-one, np.zeros((1,len(x)))])
        # Solve problem
        options=dict(tol=1e-5, autoscale=True, disp=False)
        # Set bounds on t level only
        bounds = [(None,opts['level'])] + [(None,None)]*len(x)
        res = linprog(c=c.squeeze(),
                      A_eq=A_eq,
                      b_eq=b_eq.squeeze(),
                      A_ub=A_cut,
                      b_ub=b_cut.squeeze(),
                      method='highs',
                      bounds=bounds,
                      options=options)
        if res.success:
            t_opt, x_opt = res.x[0],res.x[1:]
            x_opt = np.expand_dims(x_opt,axis=1)
        else:
            raise ValueError("Linear subproblem failed.")
        # STATUS UPDATE
        # update upper bound
        if t_opt <= t_max: 
            t_max = t_opt
        # define gap
        gap = (t_max-t_min)/np.abs(t_min)
        # termination criteria
        if t_min >= -opts['tol_eig']:
            status = "POS_LB"
        elif t_max < -2*opts['tol_eig']:
            status = "NEG_UB"
        # Update vars
        n_iter += 1
        delta_x = x_opt - x
        delta_sqr = (delta_x.T@delta_x)[0,0]
        x = x_opt
        # PRINT
        if verbose:
            if header_printed is False:
                print(' N   | delta_sqr |   t_max   |   t_min     |   gap    |')
                header_printed=True
            print(f" {n_iter:3d} | {delta_sqr:5.4e} | {t_max:5.4e} | {t_min:5.4e} | {gap:5.4e}")
    
    # Set outputs
    output = dict(H=H,
                  status=status,
                  mults=x,
                  gap=gap,
                  t_min=t_min,
                  t_max=t_max,
                )
    
    return output
    
def cvxpy_qp(P, q, G, h, A=None, b=None, verbose=False):
    x = cp.Variable((P.shape[0],1),'x')
    eps = cp.Variable((A.shape[0],1),'eps')
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + 2*q.T @ x + 500*cp.quad_form(eps,np.eye(eps.shape[0]))),
                    [G @ x <= h,
                    A @ x - b == eps])
    options = dict(feastol=1e-5,reltol=1e-5,abstol=1e-5)
    prob.solve(verbose=verbose, solver='CVXOPT', **options)
    return x.value

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
        