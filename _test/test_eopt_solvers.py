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
    res = get_grad_info(H,A,k=4,method="direct")
    subgrad, min_eig,hessian,t = res['subgrad'],res['min_eig'],res['hessian'],res['multplct']
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
    res = get_grad_info(H,A,k=4,method="direct")
    subgrad, min_eig, hessian, t = res['subgrad'],res['min_eig'],res['hessian'],res['multplct']
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
    tol = 1e-6
    input = dict(A_list=A,k=4,method="direct",tol=eps**2,get_hessian=True)
    H_00 = H
    H_10 = H+eps*A[0]
    H_01 = H+eps*A[1]
    res = get_grad_info(H_00,**input)
    grad_eps00, min_eig_eps00, hessian_eps00 = res['subgrad'],res['min_eig'],res['hessian']
    res = get_grad_info(H_10,**input)
    grad_eps10, min_eig_eps10, hessian_eps10 = res['subgrad'],res['min_eig'],res['hessian']
    res = get_grad_info(H_01,**input)
    grad_eps01, min_eig_eps01, hessian_eps01 = res['subgrad'],res['min_eig'],res['hessian']
    # Check gradient
    grad_num = np.vstack([(min_eig_eps10 - min_eig_eps00)/eps,
                          (min_eig_eps01 - min_eig_eps00)/eps])
    np.testing.assert_allclose(grad_eps00,
                                grad_num,
                                atol=tol,
                                rtol=0,
                                err_msg="Computed gradient does not match numerical.")
    # Check Hessian
    hessian_num = np.hstack([(grad_eps10 - grad_eps00)/eps,
                             (grad_eps01 - grad_eps00)/eps])
    np.testing.assert_allclose(hessian_eps00,
                                hessian_num,
                                atol=tol,
                                rtol=0,
                                err_msg="Computed hessian does not match numerical.")
    # Taylor Expansion Check 1
    delta = eps*np.array([[1],[0]])
    eig_delta_taylor = 1/2 * delta.T @ hessian_eps00 @ delta + grad_eps00.T@delta
    eig_delta = min_eig_eps10 - min_eig_eps00
    np.testing.assert_allclose(eig_delta_taylor,eig_delta,atol=0, rtol=1e-5)
    # Taylor Expansion Check 2
    delta = eps*np.array([[0],[1]])
    eig_delta_taylor = 1/2 * delta.T @ hessian_eps00 @ delta + grad_eps00.T@delta
    eig_delta = min_eig_eps01 - min_eig_eps00
    np.testing.assert_allclose(eig_delta_taylor,eig_delta,atol=0, rtol=1e-5)
        
def test_qp_subproblem():
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
    A_list = []
    A_list += [sp.diags([1.,0.,0.,0.])]
    A_list += [sp.diags([0.,1.,0.,0.])]
    A_list += [sp.diags([0.,0.,1.,0.])]
    A_list += [sp.diags([0.,0.,0.,1.])]
    # Compute Gradient
    grad_info = get_grad_info(H,A_list,k=4,method="direct",get_hessian=True)
    # Compute QP solution with no constraints
    A_qp = sp.csc_array((1,4))
    b_qp = 0.
    step, cost_delta = solve_step_qp(grad_info=grad_info, A=A_qp, b=b_qp)
    # Apply step
    H1 = H + np.sum([x * A for (x,A) in zip(step[:,0].tolist(), A_list)])
    # Check new minimum eigenvalue
    grad_info1 = get_grad_info(H1,A_list,k=4,method="direct")
    delta_min_eig = grad_info1['min_eig'] -  grad_info['min_eig']
    np.testing.assert_almost_equal(cost_delta,delta_min_eig, decimal=6)
    
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
    
def run_eopt_cuts(prob_file="test_prob_1.pkl"):
    # Test SQP method
    # Load data from file
    with open(os.path.join(root_dir,"_test",prob_file),'rb') as file:
        data = pickle.load(file)
    # Get global solution
    u,s,v = np.linalg.svd(data['X'])
    x_0 = u[:,[0]] * np.sqrt(s[0])
    # Run optimizer
    output = solve_eopt_cuts(C=data['Q'],
                            Constraints=data['Constraints'],
                            x_cand=x_0,
                            ) 
    
def test_eopt_cuts_poly():    
    # Run Optimizer
    solve_eopt_cuts(**poly6_test())

def poly6_test():
    # Sixth order polynomial that requires redundant constraints to solve
    Q = np.array([[5.0000,1.3167,-1.4481,0],
                   [1.3167,-1.4481,0,0.2685],
                   [-1.4481,0,0.2685,-0.0667],
                   [0,0.2685,-0.0667,0.0389]])
    Constraints=[]
    A = np.zeros((4,4)) # w^2 = 1
    A[0,0] = 1
    Constraints += [(A, 1.)]
    A = np.zeros((4,4)) # x^2 = x*x
    A[2,0] = 1/2
    A[0,2] = 1/2
    A[1,1] = -1
    Constraints += [(A, 0.)]
    A = np.zeros((4,4)) # x^3 = x^2*x
    A[3,0] = 1
    A[0,3] = 1
    A[1,2] = -1
    A[2,1] = -1
    Constraints += [(A, 0.)]
    A = np.zeros((4,4)) # x^3*x = x^2*x^2
    A[3,1] = 1/2
    A[1,3] = 1/2
    A[2,2] = -1 
    Constraints += [(A, 0.)]
    # Candidate solution
    x_cand = np.array([[1.0000,-1.4871,2.2115,-3.2888]]).T
    # Dual optimal
    mults = -np.array([[-3.1937],
                    [ 2.5759],
                    [-0.0562],
                    [ 0.8318]])

    # output
    output = dict(Constraints=Constraints,
                  C=Q,
                  x_cand=x_cand,
                  opt_mults = mults)
    return output
    
def test_eopt_project():
    run_eopt_project(prob_file="test_prob_4.pkl")

def test_eopt_sqp():
    run_eopt_sqp(prob_file="test_prob_7.pkl")
    
def test_eopt_cuts():
    run_eopt_cuts(prob_file="test_prob_7.pkl")
        
if __name__ == "__main__":
    # GRADIENT TESTS
    # test_subgradient_analytic()
    # test_subgradient_mult2()
    # test_grad_hess_numerical()
    
    # QUADRATIC SUBPROBLEM TESTS
    # test_qp_subproblem()
    
    # EOPT TESTS
    # test_eopt_project()
    # test_eopt_penalty()
    # test_eopt_sqp()
    test_eopt_cuts()
    # test_eopt_cuts_poly()