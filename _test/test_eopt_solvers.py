from cert_tools.eopt_solvers import *
# Maths
import numpy as np
import scipy.sparse as sp
# Data
import pickle
# System
import os, sys
from os.path import dirname
# Test file
from cert_tools.eopt_solvers import solve_eopt_penalty

sys.path.append(dirname(__file__) + "/../")
root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")

def test_subgradient_analytic():
    # Define eigenvalues and vectors
    eig_vals = [-1., 1., 1., 3.]
    D = np.diag(eig_vals)
    T = np.random.rand(4,4)*2-1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    H = Q @ D @ Q.T
    # Constraint matrices
    A = []
    A += [sp.diags([1.,0.,0.,0.])]
    A += [sp.diags([0.,1.,0.,0.])]
    # Compute subgrad and actual subgrad (with default U)
    subgrad, min_eig,hessian,t = get_subgradient(H,A,k=4,method="direct")
    subgrad_true =np.array([ Q[0,0]**2, Q[1,0]**2 ])
    # Check length
    assert len(subgrad) == len(A), \
        ValueError("Subgradient should have length equal to that of constraints")
    # Check multiplicity
    assert t == 1, "Multiplicity is incorrect"
    # Check eig
    np.testing.assert_almost_equal(min_eig, -1.)
    # Check subgradient
    np.testing.assert_allclose(subgrad,subgrad_true, rtol=0 , atol=1e-8)
    
def test_grad_hess_numerical():
    np.random.seed(0)
    # Define eigenvalues and vectors
    eig_vals = [-1., 1., 1., 3.]
    D = np.diag(eig_vals)
    T = np.random.rand(4,4)*2-1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    H = Q @ D @ Q.T
    # Constraint matrices
    A = []
    A += [sp.diags([1.,0.,0.,0.])]
    A += [sp.diags([0.,1.,0.,0.])]
    # Compute subgrad and min eigs for first difference
    eps = 1e-8
    grad_eps00, min_eig_eps00, hessian_eps00, _ = get_subgradient(H,A,k=4,method="direct",tol=eps)
    grad_eps10, min_eig_eps10, hessian_eps10, _ = get_subgradient(H+eps*A[0],A,k=4,method="direct",tol=eps)
    grad_eps01, min_eig_eps01, hessian_eps01, _ = get_subgradient(H+eps*A[1],A,k=4,method="direct",tol=eps)
    grad_eps11, min_eig_eps11, hessian_eps11, _ = get_subgradient(H+eps*A[0]+eps*A[1],A,k=4,method="direct",tol=eps)
    # Check gradient
    grad_num = np.vstack([(min_eig_eps10 - min_eig_eps00)/eps,
                          (min_eig_eps01 - min_eig_eps00)/eps])
    np.testing.assert_allclose(grad_eps00,
                                grad_num,
                                atol=1e-5,
                                rtol=0,
                                err_msg="Computed gradient does not match numerical.")
    # Check Hessian
    hessian_num = np.hstack([(grad_eps10 - grad_eps00)/eps,
                             (grad_eps01 - grad_eps00)/eps])
    np.testing.assert_allclose(hessian_eps00,
                                hessian_num,
                                atol=1e-5,
                                rtol=0,
                                err_msg="Computed hessian does not match numerical.")
            
def test_subgradient_mult2():
    "Multiplicity 2 test"
    # Define eigenvalues and vectors 
    eig_vals = [-1., -1., 1., 3.]
    D = np.diag(eig_vals)
    T = np.random.rand(4,4)*2-1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    H = Q @ D @ Q.T
    # Constraint matrices
    A = []
    A += [sp.diags([1.,0.,0.,0.])]
    A += [sp.diags([0.,1.,0.,0.])]
    # Compute subgrad and actual subgrad (with default U)
    subgrad, min_eig, hessian, t = get_subgradient(H,A,k=4,method="direct")
    subgrad_true =np.array([ Q[0,0]**2 + Q[0,1]**2, Q[1,0]**2 + Q[1,1]**2 ])/2.
    # Check length
    assert len(subgrad) == len(A), \
        ValueError("Subgradient should have length equal to that of constraints")
    # Check multiplicity
    assert t == 2, "Multiplicity is incorrect"
    # Check eig
    np.testing.assert_almost_equal(min_eig, np.min(eig_vals))
    # Check subgradient
    np.testing.assert_allclose(subgrad,
                               subgrad_true,
                               rtol=0 ,
                               atol=1e-8)

def test_eopt_penalty(prob_file="test_prob_1.pkl"):
    # Test penalty method
    # Load data from file
    with open(os.path.join(root_dir,"_test",prob_file),'rb') as file:
        data = pickle.load(file)
    # Get global solution
    u,s,v = np.linalg.svd(data['X'])
    x_0 = u[:,[0]] * np.sqrt(s[0])
    # Run optimizer
    H, info = solve_eopt_penalty(Q=data['Q'],
                        Constraints=data['Constraints'],
                        x_cand=x_0
                        )

def run_eopt_project(prob_file="test_prob_1.pkl"):
    # Test penalty method
    # Load data from file
    with open(os.path.join(root_dir,"_test",prob_file),'rb') as file:
        data = pickle.load(file)
    # Get global solution
    u,s,v = np.linalg.svd(data['X'])
    x_0 = u[:,[0]] * np.sqrt(s[0])
    # Run optimizer
    H, info = solve_eopt_project(Q=data['Q'],
                                Constraints=data['Constraints'],
                                x_cand=x_0,
                                )

def run_eopt_sqp(prob_file="test_prob_1.pkl"):
    # Test SQP method
    # Load data from file
    with open(os.path.join(root_dir,"_test",prob_file),'rb') as file:
        data = pickle.load(file)
    # Get global solution
    u,s,v = np.linalg.svd(data['X'])
    x_0 = u[:,[0]] * np.sqrt(s[0])
    # Run optimizer
    H, info = solve_eopt_sqp(Q=data['Q'],
                            Constraints=data['Constraints'],
                            x_cand=x_0,
                            )

def test_eopt_project():
    run_eopt_project(prob_file="test_prob_4.pkl")

def test_eopt_sqp():
    run_eopt_sqp(prob_file="test_prob_6.pkl")
    
if __name__ == "__main__":
    # test_subgradient_analytic()
    # test_subgradient_mult2()
    # test_grad_hess_numerical()
    # test_eopt_project()
    # test_eopt_penalty()
    test_eopt_sqp()