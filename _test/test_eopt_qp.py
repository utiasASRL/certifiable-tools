import numpy as np


def test_eopt_qp():
    """
    Use test problem from Overton 1988
    """
    from cert_tools import solve_Eopt_QP, f_Eopt

    Q = np.eye(2)
    A1 = np.r_[np.c_[1.0, 0.0], np.c_[0.0, -1.0]]
    A2 = lambda kappa: np.r_[np.c_[1, kappa], np.c_[kappa, 4]]

    # for large kappa, problem should be easy.
    # this is the example from Overton 1988, for which we know the exact solution.
    kappa = 3.0
    A_list = [A1, A2(kappa)]
    x_init = [1.0, 2.0]

    l_max = f_Eopt(Q, A_list, x_init)
    assert abs(l_max - 12.32) < 0.01

    x_sol, info = solve_Eopt_QP(Q, A_list, x_init, verbose=2)
    U = info["U"]
    np.testing.assert_allclose(x_sol, 0.0, rtol=1e-7, atol=1e-8)
    if U is not None:
        np.testing.assert_allclose(
            U, np.r_[np.c_[0.5, -5 / (4 * kappa)], np.c_[-5 / (4 * kappa), 0.5]]
        )

    print("big kappa test passed")

    # for small kappa, problem is unbounded. Most importantly, as stated in
    # Overton 1988, p. 264, there is a valid descent direction at (0, 0)
    kappa = 2.25
    A_list = [A1, A2(kappa)]
    x_init = [0.0, 0.0]
    x_sol, info = solve_Eopt_QP(Q, A_list, x_init, verbose=2)
    assert info["success"] == False
    print("small kappa test passed")


if __name__ == "__main__":
    test_eopt_qp()
    print("all tests passed")
