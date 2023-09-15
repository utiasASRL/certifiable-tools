# Optimization
import mosek
import cvxpy as cp

# Maths
import numpy as np
import scipy.sparse as sp
from cert_tools.eig_tools import get_min_eigpairs


def get_subgradient(H, A_list, U=None, tau=1e-8, **kwargs):
    eig_vals, eig_vecs = get_min_eigpairs(H, **kwargs)
    # get minimum eigenvalue     
    min_eig = np.min(eig_vals)
    # get minimum eigenvectors (multiplicity could be > 1)
    inds = np.abs(eig_vals - min_eig) < tau
    Q = eig_vecs[:,inds]
    # Scale variable
    if U is None:
        U = 1 / Q.shape[1] * np.eye(Q.shape[1])
    # Compute gradient
    subgrad = [np.trace(U @ (Q.T @ Ai @ Q)) for Ai in A_list]
    return subgrad, min_eig

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
        subgrad = get_subgradient(Q, A_list, a)

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

def solve_eopt_penalty(Q, Constraints, x_cand, opts, verbose=True):
    """Solve the certificate/eigenvalue optimization problem using a penalty term
    to enforce the affine constraints (i.e. first order opt conditions). """
    
    # Get affine constraint from constraints and candidate solution
    A_bar = sp.hstack([A @ x_cand for A,b in Constraints])
    b_bar = Q @ x_cand
    # TODO Check sign here ^^^
    # Initialize
    min_eig = -np.inf
    aff_cost = inf
    n_iter = 0
    x = np.zeros(A_bar.shape[1])
    while (min_eig < -opts['tol_eig'] or aff_cost > opts['tol_aff']) and \
        n_iter < opts['max_iter']:
        # Construct Certificate matrix
        H_curr = Q + np.sum([ai * Ai for ai, (Ai,b) in zip(x,Constraints)])
        # Compute subgradient of ei  
