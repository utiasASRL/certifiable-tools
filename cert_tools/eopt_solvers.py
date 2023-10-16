# Optimization
import mosek
import cvxpy as cp
from scipy.optimize import linprog
import scipy
import gurobipy as gp
import quadprog

# Maths
import numpy as np
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
opts_cut_dflt = dict(tol_eig = 1e-6,
                     max_iter=1000,
                     method = 'level',
                     min_eig_ub = 0.0,
                     lambda_level = 0.95,
                     level_method_bound =1e4,
                     tol_affine = 1e-5,
                     tol_relgap = 1e-8)


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
            if n_iter>0:
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

# @profile
def solve_eopt_cuts(C, Constraints, x_cand, opts=opts_cut_dflt, verbose=True,**kwargs):
    """Solve the certificate/eigenvalue optimization problem using a cutting plane algorithm.
    Current algorithm uses the level method with the target level at a tolerance below zero"""
    # Get affine constraint from constraints and candidate solution
    A_bar = np.hstack([A @ x_cand for A,b in Constraints])
    b_bar = -C @ x_cand
    A_list = [A for A,b in Constraints]
    A_vec = sp.hstack([A.reshape((-1,1),order='F') for A,b in Constraints])
    # Truncate eigenvalues of A_bar to make nullspace more defined
    Q,R,P = scipy.linalg.qr(A_bar, pivoting=True)
    for i,r in enumerate(np.diag(R)):
        if np.abs(r) < opts['tol_affine']:
            i-=1
            break
    Q1 = Q[:,:i+1]
    R1 = R[:i+1,:]
    # Define affine constraints in epigraph form (with col of zeros)
    Pinv = np.zeros(len(P),int)
    for k,p in enumerate(P):
        Pinv[p] = k
    A_eq = np.hstack([np.zeros((R1.shape[0], 1)),R1[:,Pinv]])
    b_eq = Q1.T @ b_bar
    # Get orthogonal vector to x_cand
    x_rand = np.random.rand(*x_cand.shape)-1
    x_orth = x_rand - x_cand.T@x_rand / (x_cand.T@x_cand)
    # INITIALIZE
    status = 'RUNNING'
    header_printed=False
    n_iter = 0
    t_max = np.inf
    t_min = -np.inf
    iter_info = []
    # Initialize multiplier variables
    #x = np.linalg.lstsq(R1, b_eq)[0]
    x = np.zeros((len(Constraints),1))
    # x = kwargs['opt_mults']
    # Init cutting plane constraints 
    #    t <= level
    # <==>   [ 1 0 ]@[t;x] <= level
    one = np.array([[1.]])
    A_cut = None
    b_cut = None
    # LOOP
    while status == "RUNNING":
        # SOLVE CUT PLANE PROGRAM WITH MAX LEVEL
        # Cost: maximize t
        c = np.hstack([-one, np.zeros((1,len(x)))])
        # Set bounds on t level only
        bounds = [(None,opts['min_eig_ub'])] + [(None,None)]*len(x)
        # Solve opt problem
        res = linprog(c=c.squeeze(),
                      A_eq=A_eq,
                      b_eq=b_eq.squeeze(),
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
                                A_eq=A_eq,
                                b_eq=b_eq,
                                A_ub=A_cut,
                                b_ub=b_cut,)
            t_qp, x_new = t_x[0,:],t_x[1:,:]
        else:
            x_new = x_lp
        # CUT PLANE UPDATE
        # Construct Current Certificate matrix
        H = get_cert_mat(C, A_vec, x_new)
        # Number of eigenvalues to compute
        k = 10
        # current gradient and minimum eig
        grad_info = get_grad_info(H=H, A_list=A_list, k=k,method="direct",v0=x_orth)
        if grad_info['min_eig'] > t_min:
            t_min = grad_info['min_eig']
        # Add Cuts
        # NOTE: t <= grad @ (x-x_bar) + f(x_bar) <==> 
        # [1 -grad] @ [t; x] <= f(x_bar) -grad @ x_bar  
        a_cut = np.hstack([one, -grad_info['subgrad'].T])
        b_val = grad_info['min_eig'] - grad_info['subgrad'].T @ x_new
        if A_cut is None:
            A_cut = a_cut
            b_cut = b_val
        else:
            A_cut = np.vstack([A_cut, a_cut])
            b_cut = np.hstack([b_cut, b_val])
        # STATUS UPDATE
        # update model upper bound
        if t_lp <= t_max: 
            t_max = t_lp
        # define gap
        gap = (t_max-t_min)
        # termination criteria
        if gap <= opts['tol_relgap'] and False:# Off for now
            status = "REL_GAP"
        elif t_min >= -opts['tol_eig']:
            status = "POS_LB"
        elif t_max < -2*opts['tol_eig']:
            status = "NEG_UB"
        elif n_iter >= opts['max_iter']:
            status = "MAX_ITER"

        # Update vars
        n_iter += 1
        delta_x = x_new - x
        delta_norm = np.linalg.norm(delta_x)
        x = x_new
        
        # PRINT
        if verbose:
            if n_iter % 10 == 1:
                header_printed=False
            if header_printed is False:
                print(' N   | delta_nrm |  eig val  |   t_max   |   t_min     |   gap    | mult. |')
                header_printed=True
            print(f" {n_iter:3d} | {delta_norm:5.4e} | {grad_info['min_eig']:5.4e} | {t_max:5.4e} | {t_min:5.4e} | {gap:5.4e} | {grad_info['multplct']:4d} ")
            # Store data
            info=dict(n_iter=n_iter,
                    delta_norm=delta_norm,
                    min_eig_curr=grad_info['min_eig'],
                    t_max=t_max,
                    t_min=t_min,
                    gap=gap,
                    mult=grad_info['multplct']
                    )
            iter_info+=[info]
            
            
    # Set outputs
    output = dict(H=H,
                  status=status,
                  mults=x,
                  gap=gap,
                  t_min=t_min,
                  t_max=t_max,
                  iter_info=pd.DataFrame(iter_info)
                )
    
    return output

def level_project(x_prox, level, A_eq, b_eq, A_ub, b_ub):
    """Solve level set projection problem: return the closest point to x_prox 
    (the prox-center) such that the acheived model value is above the provided
    level"""
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
    # Solve Quadratic Program:
    x = cp.Variable((len(x_prox)+1,1),'x')
    prob = cp.Problem(cp.Minimize(cp.norm2(x[1:,:]-x_prox)),
                    [G @ x <= h,
                    A_eq @ x == b_eq])
    prob.solve(solver='MOSEK', verbose=False)
    # Get solution
    return x.value
    
def get_cert_mat(C, A_vec, mults, sparsify=True):
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
    # Update H
    H += (A_vec @ mults).reshape(H.shape,order='F')
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

