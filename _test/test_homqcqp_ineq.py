import numpy as np
import pytest
from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.sdp_solvers import solve_sdp_homqcqp
from cert_tools.sparse_solvers import solve_dsdp


def random_symmetric(d):
    A = np.random.rand(d, d)
    return 0.5 * (A + A.T)


def test_obstacle():
    """
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

    C = PolyMatrix()
    C["h", "h"] = x_s.T @ x_s + x_t.T @ x_t
    C["h", f"x1"] = -x_s.T
    C["h", f"x3"] = -x_t.T
    C["x1", "x1"] = 2 * np.eye(d)
    C["x1", "x2"] = -np.eye(d)
    C["x2", "x2"] = 2 * np.eye(d)
    C["x2", "x3"] = -np.eye(d)
    C["x3", "x3"] = 2 * np.eye(d)

    A_list = []
    B_list = []
    for i in range(1, N + 1):
        B = PolyMatrix()
        B[f"x{i}", f"x{i}"] = np.eye(d)
        B[f"h", f"x{i}"] = -o.T
        B["h", "h"] = o.T @ o - 4
        B_list.append(B)

    problem = HomQCQP.create_from_matrices(C, A_list, B_list)

    with pytest.raises(AssertionError):
        X, info = solve_dsdp(problem)

    problem.clique_decomposition(
        clique_data=[["h", f"x{i}", f"x{i+1}"] for i in range(1, N)]
    )
    c_list, info_dsdp = solve_dsdp(problem)

    Y, ranks, factors = problem.get_mr_completion(c_list)
    print("done")
    Y_gt = np.array([1, 0, 1, 0, 2, 0, 3])
    np.testing.assert_allclose(Y.flatten(), Y_gt, rtol=1e-3, atol=1e-3)

    X, info_full = solve_sdp_homqcqp(problem)
    np.testing.assert_allclose(X, Y @ Y.T, rtol=1e-2, atol=1e-2)
    assert abs(info_full["cost"] - info_dsdp["cost"]) < 1e-4


if __name__ == "__main__":
    test_obstacle()
