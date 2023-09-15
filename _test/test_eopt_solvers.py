from cert_tools.eopt_solvers import *
# Maths
import numpy as np
import scipy.sparse as sp

def test_subgradient():
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
    subgrad, min_eig = get_subgradient(H,A,k=4,method="direct")
    subgrad_true =np.array([ Q[0,0]**2, Q[1,0]**2 ])
    # Check length
    assert len(subgrad) == len(A), \
        ValueError("Subgradient should have length equal to that of constraints")
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
    subgrad, min_eig = get_subgradient(H,A,k=4,method="direct")
    subgrad_true =np.array([ Q[0,0]**2 + Q[0,1]**2, Q[1,0]**2 + Q[1,1]**2 ])/2.
    # Check length
    assert len(subgrad) == len(A), \
        ValueError("Subgradient should have length equal to that of constraints")
    # Check eig
    np.testing.assert_almost_equal(min_eig, np.min(eig_vals))
    # Check subgradient
    np.testing.assert_allclose(subgrad,subgrad_true, rtol=0 , atol=1e-8)
    
if __name__ == "__main__":
    test_subgradient()
    test_subgradient_mult2()