import matplotlib.pylab as plt
import numpy as np
import pytest
from matplotlib.patches import Circle
from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.sdp_solvers import solve_feasibility_sdp, solve_sdp_homqcqp
from cert_tools.sparse_solvers import (
    solve_dsdp,
    solve_feasibility_dsdp,
    solve_feasibility_dsdp_fusion,
)

PLOT = False


def mat_test_and_plot(H1, H2):
    try:
        np.testing.assert_almost_equal(H1, H2, decimal=2)
    except AssertionError:
        fig, axs = plt.subplots(1, 3)
        axs[0].matshow(H1)
        axs[1].matshow(H2)
        axs[2].matshow(H1 - H2)
        plt.show()
        raise


def random_symmetric(d):
    A = np.random.rand(d, d)
    return 0.5 * (A + A.T)


def test_obstacle():
    """
        -1   0   1
            goal
    4        o
        _------ _
    3            \
                  \
    2    x        |
                  /
    1            /
        ------""
    0        o
            start

    go from (0, 0) to (0, 4) avoiding obstacle of radius 2 at (-1, 2)
            --xs--    --xt--                             r     --o---
    min f(x) := ||x_1 - x_s||^2 + ||x_2 - x_1||^2 + ||x_3 - x_2||^2 + ||x_t - x_3||^2
        s.t. g_i(x) :=|| x_i - o ||^2 - 4 >= 0

    f(x) = x_s^2 + x_t^2 + x_1^2 - 2x_s'x_1 + x_2^2 - 2x_1'x_2  + x_1^2 + x_2^2 - 2x_t'x_2
         = x_s^2 + x_t^2 + 2x_1^2 -2x_s'x_1 + 2x_2^2 - 2x_1'x_s - 2x_t'x_2

    g_i(x) = o^2 - 4 - 2x_i'o + x_i^2

    """
    d = 2
    N = 3
    x_s = np.array([0, 0])[:, None]
    x_t = np.array([0, 4])[:, None]
    o = np.array([-1, 2])[:, None]
    radius = 2

    C = PolyMatrix()
    C["h", "h"] = x_s.T @ x_s + x_t.T @ x_t
    C["h", f"x1"] = -x_s.T
    C["h", f"x3"] = -x_t.T
    C["x1", "x1"] = 2 * np.eye(d)
    C["x1", "x2"] = -np.eye(d)
    C["x2", "x2"] = 2 * np.eye(d)
    C["x2", "x3"] = -np.eye(d)
    C["x3", "x3"] = 2 * np.eye(d)
    # offset = C["h", "h"]
    # C["h", "h"] -= offset

    A_list = []
    B_list = []
    for i in range(1, N + 1):
        B = PolyMatrix()
        # r^2 - ||o_k - x_i||^2 <= 0
        B[f"x{i}", f"x{i}"] = -np.eye(d)
        B[f"h", f"x{i}"] = +o.T
        B["h", "h"] = -o.T @ o + radius**2
        B_list.append(B)

    problem = HomQCQP.create_from_matrices(C, A_list, B_list)
    problem.clique_decomposition(
        clique_data=[["h", f"x{i}", f"x{i+1}"] for i in range(1, N)]
    )
    if PLOT:
        fig, axs = plt.subplots(1, 1 + len(A_list) + len(B_list), squeeze=False)
        axs[0, 0].matshow(C.get_matrix(problem.var_sizes).toarray())
        i = 0
        for A in A_list:
            i += 1
            axs[0, i].matshow(A.get_matrix(problem.var_sizes).toarray())

        for B in B_list:
            i += 1
            axs[0, i].matshow(B.get_matrix(problem.var_sizes).toarray())
        plt.show()

    c_list, info_dsdp = solve_dsdp(problem)

    Y, ranks, factors = problem.get_mr_completion(c_list)
    print("done")
    Y_gt = np.array([1.0, 0.664, 0.891, 1.0, 2.0, 0.664, 3.109])
    np.testing.assert_allclose(Y.flatten(), Y_gt, rtol=1e-3, atol=1e-3)

    X, info_full = solve_sdp_homqcqp(problem)
    np.testing.assert_allclose(X, Y @ Y.T, rtol=1e-2, atol=1e-2)
    assert abs(info_full["cost"] - info_dsdp["cost"]) < 1e-4

    if PLOT:
        traj = Y[1:].reshape((-1, 2))
        fig, ax = plt.subplots()
        circ = Circle(xy=o, radius=radius, color="gray")
        ax.add_patch(circ)
        ax.scatter(*traj.T)
        ax.scatter(*x_s)
        ax.scatter(*x_t)
        ax.axis("equal")
        plt.show()

    H_dsdp = problem.get_dual_matrix(info_dsdp["dual"], var_list=problem.var_list)
    assert np.max(np.abs(H_dsdp @ Y_gt)) <= 1e-2

    ## create certificate
    C, Constraints, B_list = problem.get_problem_matrices()

    # 1. from canddiate solution
    nu_0 = -info_dsdp["cost"]
    x_cand = Y
    L = np.hstack([A @ x_cand for A, b in Constraints] + [B @ x_cand for B in B_list])
    b = -C @ x_cand.flatten()
    y, *_ = np.linalg.lstsq(L, b, rcond=-1)

    # 2. using feasibility SDP
    H, info_H = solve_feasibility_sdp(
        C,
        Constraints=Constraints,
        B_list=B_list,
        x_cand=Y,
        nu_0=nu_0,
        soft_epsilon=True,
        eps_tol=1e-4,
        adjust=False,
    )
    assert abs(info_H["yvals"][0] - nu_0) < 1e-3
    assert H is not None
    np.testing.assert_allclose(y[1:], info_H["mus"], rtol=1e-3)

    H_poly, __ = PolyMatrix.init_from_sparse(H, problem.var_sizes, symmetric=False)
    c_list_new, info_H_dsdp = solve_feasibility_dsdp(
        problem,
        x_cand=Y,
        nu_0=nu_0,
        tol=1e-4,
        soft_epsilon=False,
        eps_tol=1e-3,
        test_H_poly=H_poly,
    )
    print("epsilon:", info_H_dsdp["epsilon"])
    np.testing.assert_allclose(y[1:], info_H_dsdp["mus"], rtol=1e-3)
    H_test = problem.get_dual_matrix(c_list_new, var_list=problem.var_list)

    H_test2 = problem.get_dual_matrix_from_vars(
        info_H_dsdp["nu_0"], info_H_dsdp["yvals"], info_H_dsdp["mus"]
    )
    mat_test_and_plot(H_test.toarray(), H_test2)

    fig, axs = plt.subplots(1, len(c_list_new))
    for i, (ax, Hi) in enumerate(zip(axs, c_list_new)):
        ax.matshow(Hi, vmax=np.max(H_test), vmin=np.min(H_test))
        ax.set_title(f"block {i}")

    mat_test_and_plot(H_test.toarray(), H.toarray())

    c_list_fusion, info_fusion = solve_feasibility_dsdp_fusion(
        problem,
        x_cand=Y,
        nu_0=nu_0,
        tol=1e-4,
        soft_epsilon=False,
        eps_tol=1e-3,
        test_H_poly=H_poly,
    )
    H_test_fusion = problem.get_dual_matrix(c_list_fusion, var_list=problem.var_list)
    try:
        np.testing.assert_almost_equal(
            H_test.toarray(), H_test_fusion.toarray(), decimal=2
        )
    except AssertionError:
        fig, axs = plt.subplots(1, 3)
        axs[0].matshow(H_test.toarray())
        axs[1].matshow(H_test_fusion.toarray())
        axs[2].matshow((H_test - H_test_fusion).toarray())
        plt.show()
    print("done")


if __name__ == "__main__":
    test_obstacle()
