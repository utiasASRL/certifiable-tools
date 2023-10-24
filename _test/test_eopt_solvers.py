from cert_tools.eopt_solvers import *

# Maths
import numpy as np
import scipy.sparse as sp

# Data
import pickle

# System
import os, sys
from os.path import dirname

sys.path.append(dirname(__file__) + "/../")
root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def test_subgradient_analytic():
    # Define eigenvalues and vectors
    eig_vals = [-1.0, 1.0, 1.0, 3.0]
    D = np.diag(eig_vals)
    T = np.random.rand(4, 4) * 2 - 1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    H = Q @ D @ Q.T
    # Constraint matrices
    A_list = []
    A_list += [sp.diags([1.0, 0.0, 0.0, 0.0])]
    A_list += [sp.diags([0.0, 1.0, 0.0, 0.0])]
    A_vec = sp.hstack([A.reshape((-1, 1), order="F") for A in A_list])
    # Compute subgrad and actual subgrad (with default U)
    res = get_grad_info(H, A_vec, k=4, method="direct")
    subgrad, min_eig, hessian, t = (
        res["subgrad"],
        res["min_eig"],
        res["hessian"],
        res["multplct"],
    )
    subgrad_true = np.array([Q[0, 0] ** 2, Q[1, 0] ** 2])
    # Check length
    assert len(subgrad) == len(A_list), ValueError(
        "Subgradient should have length equal to that of constraints"
    )
    # Check multiplicity
    assert t == 1, "Multiplicity is incorrect"
    # Check eig
    np.testing.assert_almost_equal(min_eig, -1.0)
    # Check subgradient
    np.testing.assert_allclose(subgrad, subgrad_true, rtol=0, atol=1e-8)


def test_subgradient_mult2():
    "Multiplicity 2 test"
    # Define eigenvalues and vectors
    eig_vals = [-1.0, -1.0, 1.0, 3.0]
    D = np.diag(eig_vals)
    T = np.random.rand(4, 4) * 2 - 1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    H = Q @ D @ Q.T
    # Constraint matrices
    A_list = []
    A_list += [sp.diags([1.0, 0.0, 0.0, 0.0])]
    A_list += [sp.diags([0.0, 1.0, 0.0, 0.0])]
    A_vec = sp.hstack([A.reshape((-1, 1), order="F") for A in A_list])
    # Compute subgrad and actual subgrad (with default U)
    res = get_grad_info(H, A_vec, k=4, method="direct")
    subgrad, min_eig, hessian, t = (
        res["subgrad"],
        res["min_eig"],
        res["hessian"],
        res["multplct"],
    )
    subgrad_true = (
        np.array([Q[0, 0] ** 2 + Q[0, 1] ** 2, Q[1, 0] ** 2 + Q[1, 1] ** 2]) / 2.0
    )
    # Check length
    assert len(subgrad) == len(A_list), ValueError(
        "Subgradient should have length equal to that of constraints"
    )
    # Check multiplicity
    assert t == 2, "Multiplicity is incorrect"
    # Check eig
    np.testing.assert_almost_equal(min_eig, np.min(eig_vals))
    # Check subgradient
    np.testing.assert_allclose(subgrad, subgrad_true, rtol=0, atol=1e-8)


def test_grad_hess_numerical():
    np.random.seed(0)
    # Define eigenvalues and vectors
    eig_vals = [-1.0, 1.0, 1.0, 3.0]
    D = np.diag(eig_vals)
    T = np.random.rand(4, 4) * 2 - 1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    H = Q @ D @ Q.T
    # Constraint matrices
    A_list = []
    A_list += [sp.diags([1.0, 0.0, 0.0, 0.0])]
    A_list += [sp.diags([0.0, 1.0, 0.0, 0.0])]
    A_vec = sp.hstack([A.reshape((-1, 1), order="F") for A in A_list])
    # Compute subgrad and min eigs for first difference
    eps = 1e-8
    tol = 1e-6
    input = dict(A_vec=A_vec, k=4, method="direct", tol=eps**2, get_hessian=True)
    H_00 = H
    H_10 = H + eps * A_list[0]
    H_01 = H + eps * A_list[1]
    res = get_grad_info(H_00, **input)
    grad_eps00, min_eig_eps00, hessian_eps00 = (
        res["subgrad"],
        res["min_eig"],
        res["hessian"],
    )
    res = get_grad_info(H_10, **input)
    grad_eps10, min_eig_eps10, hessian_eps10 = (
        res["subgrad"],
        res["min_eig"],
        res["hessian"],
    )
    res = get_grad_info(H_01, **input)
    grad_eps01, min_eig_eps01, hessian_eps01 = (
        res["subgrad"],
        res["min_eig"],
        res["hessian"],
    )
    # Check gradient
    grad_num = np.vstack(
        [(min_eig_eps10 - min_eig_eps00) / eps, (min_eig_eps01 - min_eig_eps00) / eps]
    )
    np.testing.assert_allclose(
        grad_eps00,
        grad_num,
        atol=tol,
        rtol=0,
        err_msg="Computed gradient does not match numerical.",
    )
    # Check Hessian
    hessian_num = np.hstack(
        [(grad_eps10 - grad_eps00) / eps, (grad_eps01 - grad_eps00) / eps]
    )
    np.testing.assert_allclose(
        hessian_eps00,
        hessian_num,
        atol=tol,
        rtol=0,
        err_msg="Computed hessian does not match numerical.",
    )
    # Taylor Expansion Check 1
    delta = eps * np.array([[1], [0]])
    eig_delta_taylor = 1 / 2 * delta.T @ hessian_eps00 @ delta + grad_eps00.T @ delta
    eig_delta = min_eig_eps10 - min_eig_eps00
    np.testing.assert_allclose(eig_delta_taylor, eig_delta, atol=0, rtol=1e-5)
    # Taylor Expansion Check 2
    delta = eps * np.array([[0], [1]])
    eig_delta_taylor = 1 / 2 * delta.T @ hessian_eps00 @ delta + grad_eps00.T @ delta
    eig_delta = min_eig_eps01 - min_eig_eps00
    np.testing.assert_allclose(eig_delta_taylor, eig_delta, atol=0, rtol=1e-5)


def test_qp_subproblem():
    np.random.seed(0)
    # Define eigenvalues and vectors
    eig_vals = [-1.0, 1.0, 1.0, 3.0]
    D = np.diag(eig_vals)
    T = np.random.rand(4, 4) * 2 - 1
    # Make orthogonal matrix from T
    Q, R = np.linalg.qr(T)
    # Define Test Matrix
    H = Q @ D @ Q.T
    # Constraint matrices
    A_list = []
    A_list += [sp.diags([1.0, 0.0, 0.0, 0.0])]
    A_list += [sp.diags([0.0, 1.0, 0.0, 0.0])]
    A_list += [sp.diags([0.0, 0.0, 1.0, 0.0])]
    A_list += [sp.diags([0.0, 0.0, 0.0, 1.0])]
    # Compute Gradient
    grad_info = get_grad_info(H, A_list, k=4, method="direct", get_hessian=True)
    # Compute QP solution with no constraints
    A_qp = sp.csc_array((1, 4))
    b_qp = 0.0
    step, cost_delta = solve_step_qp(grad_info=grad_info, A=A_qp, b=b_qp)
    # Apply step
    H1 = H + np.sum([x * A for (x, A) in zip(step[:, 0].tolist(), A_list)])
    # Check new minimum eigenvalue
    grad_info1 = get_grad_info(H1, A_list, k=4, method="direct")
    delta_min_eig = grad_info1["min_eig"] - grad_info["min_eig"]
    np.testing.assert_almost_equal(cost_delta, delta_min_eig, decimal=6)


def run_eopt_project(prob_file="test_prob_1.pkl"):
    # Test penalty method
    # Load data from file
    with open(os.path.join(root_dir, "_test", prob_file), "rb") as file:
        data = pickle.load(file)
    # Get global solution
    u, s, v = np.linalg.svd(data["X"])
    x_0 = u[:, [0]] * np.sqrt(s[0])
    # Run optimizer
    H, info = solve_eopt_project(
        Q=data["Q"],
        Constraints=data["Constraints"],
        x_cand=x_0,
    )


def run_eopt_cuts(prob_file="test_prob_1.pkl", opts=opts_cut_dflt, global_min=True):
    # Test SQP method
    try:
        with open(os.path.join(root_dir, "_test", prob_file), "rb") as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Skipping {prob_file} cause file not found.")
        return None

    # Get global solution
    if "x_cand" in data:
        x_cand = data["x_cand"]
    else:
        u, s, v = np.linalg.svd(data["X"])
        x_cand = u[:, [0]] * np.sqrt(s[0])

    # Run optimizer
    Q = data["Q"].copy()
    output = solve_eopt_cuts(
        Q=Q, Constraints=data["Constraints"], x_cand=x_cand, opts=opts
    )

    # Verify certificate
    H = output["H"]
    if sp.issparse(H):
        H = H.todense()
    y = H @ x_cand
    min_eig = np.min(np.linalg.eig(H)[0])

    np.testing.assert_allclose(y, 0.0, atol=5e-4, rtol=0)
    if global_min:
        assert min_eig >= -1e-6, ValueError(
            "Minimum Eigenvalue not positive at global min"
        )
    else:
        assert min_eig <= -1e-6, ValueError(
            "Minimum Eigenvalue not negative at local min"
        )
    return output


def test_eopt_cuts_poly(plot=True):
    # Get inputs
    from examples.poly6 import get_problem

    inputs = get_problem()
    output = solve_eopt_cuts(**inputs)

    if plot:
        # Plot Algorithm results
        C = inputs["Q"]
        A_vec = output["A_vec"]
        A_vec_null = output["A_vec_null"]
        mults = output["mults"]
        model = output["model"]
        x = output["x"]
        vals = output["iter_info"]["min_eig_curr"].values
        x_iter = output["iter_info"]["x"].values
        # Plot the stuff
        alpha_max = 5
        alphas = np.expand_dims(np.linspace(-alpha_max, alpha_max, 500), axis=1)
        mneigs = np.zeros(alphas.shape)
        for i in range(len(alphas)):
            # Apply step
            H_alpha = get_cert_mat(C, A_vec, mults, A_vec_null, alphas[i, :])
            # Check new minimum eigenvalue
            gi = get_grad_info(H_alpha, A_vec, k=10, method="direct")
            mneigs[i] = gi["min_eig"]
        plt.figure()
        plt.plot(alphas, mneigs, ".-r")
        # Plot Hyperplanes
        for i in range(len(model.gradients)):
            cut = model.values[i] + model.gradients[i] * (
                x + alphas - model.eval_pts[i]
            )
            plt.plot(alphas, cut)
            plt.plot(x_iter[i] - x, vals[i], ".k")

        # Plot model
        plt.figure()
        plt.plot(alphas, mneigs, ".-r")
        mvals = [model.evaluate(alphas[i, :] + x) for i in range(len(alphas))]
        mvals = np.expand_dims(np.array(mvals), axis=1)
        plt.plot(alphas, mvals, "--")
        plt.show()

    # Verify certificate
    H = output["H"]
    if sp.issparse(H):
        H = H.todense()
    y = H @ inputs["x_cand"]
    min_eig = np.min(np.linalg.eig(H)[0])
    # Error Check
    np.testing.assert_allclose(y, np.zeros(y.shape), atol=5e-4, rtol=0)
    assert min_eig >= -1e-6, ValueError("Minimum Eigenvalue not possitive")


def test_eopt_project():
    run_eopt_project(prob_file="test_prob_6.pkl")


def test_rangeonly():
    # range-only with z=x^2+y^2
    # test_eopt_cuts(prob_file="test_prob_10G.pkl", global_min=True)
    # test_eopt_cuts(prob_file="test_prob_10Gc.pkl", global_min=True)

    # test_eopt_cuts(prob_file="test_prob_10L.pkl", global_min=False)
    # test_eopt_cuts(prob_file="test_prob_10Lc.pkl", global_min=False)

    # range-only with z = [x^2, y^2, xy]
    # test_eopt_cuts(prob_file="test_prob_11G.pkl", global_min=True)
    test_eopt_cuts(prob_file="test_prob_11Gc.pkl", global_min=True)

    # test_eopt_cuts(prob_file="test_prob_11L.pkl", global_min=False)
    # test_eopt_cuts(prob_file="test_prob_11Lc.pkl", global_min=False)


def test_polynomials():
    test_eopt_cuts(prob_file="test_prob_8G.pkl", global_min=True)
    test_eopt_cuts(prob_file="test_prob_8Gc.pkl", global_min=True)

    # test on a new polynomial's local maximum
    test_eopt_cuts(prob_file="test_prob_8L1.pkl", global_min=False)
    test_eopt_cuts(prob_file="test_prob_8L1c.pkl", global_min=False)

    # test on a new polynomial's local minimum
    test_eopt_cuts(prob_file="test_prob_8L2.pkl", global_min=False)
    test_eopt_cuts(prob_file="test_prob_8L2c.pkl", global_min=False)

    # below all correspond to same polynomial
    test_eopt_cuts(prob_file="test_prob_9G.pkl", global_min=True)
    test_eopt_cuts(prob_file="test_prob_9Gc.pkl", global_min=True)

    test_eopt_cuts(prob_file="test_prob_9L.pkl", global_min=False)
    test_eopt_cuts(prob_file="test_prob_9Lc.pkl", global_min=False)


def test_eopt_cuts(prob_file="test_prob_7.pkl", global_min=True):
    from cert_tools.eopt_solvers import opts_cut_dflt

    opts = opts_cut_dflt
    opts["tol_null"] = 1e-6
    run_eopt_cuts(prob_file=prob_file, opts=opts, global_min=global_min)


if __name__ == "__main__":
    # GRADIENT TESTS
    # test_subgradient_analytic()
    # test_subgradient_mult2()
    # test_grad_hess_numerical()

    # EOPT TESTS
    # test_eopt_project()
    # test_eopt_penalty()
    # test_eopt_sqp()

    # test on a new polynomial's globoal minimum
    test_rangeonly()
