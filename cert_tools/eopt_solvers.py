# Optimization
import mosek
import cvxpy as cp

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

def solve_Eopt_QP(Q, A_list, a_init=None, max_iters=500, gtol=1e-7):
    """Solve max_alpha sigma_min (Q + sum_i (alpha_i * A_i)), using the QP algorithm
    provided by Overton 1988.
    """
    if a_init is None:
        a_init = np.zeros(len(A_list))

    alpha = 1.0

    a = a_init
    i = 0
    while i <= max_iters:
        subgrad = get_grad_info(Q, A_list, a)

        if np.linalg.norm(subgrad) < gtol:
            msg = "Converged in gradient norm"
            success = True
            break

        a += alpha * subgrad
        i += 1
        if i == max_iters:
            msg = "Reached maximum iterations"
            success = False
    info = {"success": success, "msg": msg}
    return a, info

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

# SETQUENTIAL GRADIENT DESCENT METHOD

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
        # # Get other eigvectors and eigenvalues
        ind_s = np.abs(eig_vals - min_eig) >= tau
        Q_s = eig_vecs[:,ind_s] 
        eig_inv_diffs = 1/(min_eig - eig_vals[ind_s])
        Lambda_s = np.diag(1/eig_inv_diffs)
        # Construct Hessian
        Q_bar = np.hstack([Q_s.T @ A_k @ Q_1 for A_k in A_list])
        hessian = Q_bar.T @ Lambda_s @ Q_bar * 2
        # Compute damping term for conditioning of QP
        damp = np.min(eig_inv_diffs)
    else:
        hessian = None
    grad_info = dict(subgrad=subgrad,
                     hessian=hessian,
                     min_eig=min_eig,
                     multiplicity=t,
                     damp=damp)
    return grad_info

def solve_step_qp(grad_info, A, b, use_LM=True):
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
    solution = sp.linalg.lsqr(B, c)
    x_opt = np.expand_dims(solution[0], axis=1)
    step, mult = x_opt[:hess.shape[0],[0]], x_opt[hess.shape[0]:,[0]]
    # Compute the expected delta
    cost_delta = step.T @ hess @ step /2 + grad.T @ step
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
    status = 'run'
    header_printed=False
    n_iter = 0
    # x = sp.linalg.lsqr(A_bar, b_bar.toarray(),atol=opts['tol_pen'])[0]
    # x = np.expand_dims(x,axis=1)
    x = np.zeros((A_bar.shape[1],1))
    # Opt Loop:
    while status == 'run':
        # Construct Certificate matrix
        H = Q + np.sum([ai[0] * Ai for ai, (Ai,b) in zip(x.tolist(), Constraints)])
        # Compute eigenvalue function subgrad
        k = 100
        grad_info = get_grad_info(H,
                                  A_list,
                                  get_hessian=True,
                                  method="lanczos",
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
            else:
                b=b_bar
            step, cost_delta = solve_step_qp(grad_info=grad_info,A=A_bar,b=b)
            # print output
            if verbose:
                if header_printed is False:
                    print(' N   |   cost inc   |   cost     |   alpha    | mult |')
                    header_printed=True
                print(f" {n_iter:3d} | {cost_delta:5.4e} | {cost_prev:5.4e} | {alpha_prev:5.4e} | {grad_info['multiplicity']:4d}")
            # Check termination criteria
            if n_iter>0:
                if cost >= -opts['tol_eig']:
                    status = 'opt_found'
                elif cost_delta < opts['tol_cost']:
                    status = 'grad_zero'
                elif n_iter > opts['max_iter']:
                    status = 'max_iter'
                if not status == 'run':
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
