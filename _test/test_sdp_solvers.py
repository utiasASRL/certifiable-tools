import sys, os
from os.path import dirname
import numpy as np
import pickle
import pytest
import matplotlib.pyplot as plt

sys.path.append(dirname(__file__) + "/../")
root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")
print("appended:", sys.path[-1])

from poly_matrix import PolyMatrix, sorted_dict
from cert_tools import solve_sdp_mosek, solve_low_rank_sdp

# Global test parameters
tol = 1e-8
svr_targ = 1e8


def run_mosek_solve(prob_file="test_prob_1.pkl"):
    """Utility for creating test problems"""
    # Test mosek solve on a simple problem
    # Load data from file
    with open(os.path.join(root_dir, "_test", prob_file), "rb") as file:
        data = pickle.load(file)

    # Run mosek solver
    X, info = solve_sdp_mosek(Q=data["Q"], Constraints=data["Constraints"])
    cost = info["cost"]
    # Get singular values
    u, s, v = np.linalg.svd(X)
    print(f"SVR:  {s[0]/s[1]}")
    print(f"Cost: {cost} ")

    with open(os.path.join(root_dir, "_test", prob_file), "wb") as file:
        data["X"] = X
        data["cost"] = cost
        pickle.dump(data, file)


def low_rank_solve(prob_file="test_prob_1.pkl", rank=2):
    # Test mosek solve on a simple problem
    # Load data from file
    with open(os.path.join(root_dir, "_test", prob_file), "rb") as file:
        data = pickle.load(file)

    # Feasible initial condition
    # y_0 = [1.,1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.]
    # Y_0 = y_0 * rank
    # Init with solution
    u, s, v = np.linalg.svd(data["X"])
    Y_0 = u[:, :rank] * np.sqrt(s[:rank])
    Y_0 = Y_0.reshape((-1, 1), order="F")
    Y, info = solve_low_rank_sdp(
        Q=data["Q"], Constraints=data["Constraints"], rank=rank, x_cand=Y_0
    )
    X = info["X"]
    H = info["H"]
    cost = info["cost"]

    # Check solution rank
    u, s, v = np.linalg.svd(X)
    svr = s[0] / s[1]
    cost_targ = data["cost"]
    print(f"SVR:  {svr}")
    print(f"Cost: {cost} ")
    print(f"Target Cost: {cost_targ}")
    # Minimum Eigenvalue of Certificate
    min_eig = np.min(np.linalg.eigvalsh(H.todense()))
    print(f"Minimum Certificate Eig: {min_eig}")
    # plt.semilogy(s)
    # plt.title(f"Rank Restriction: {rank}")
    # plt.show()

    return min_eig, svr, cost, cost_targ


def low_rank_test(**kwargs):
    min_eig, svr, cost, cost_targ = low_rank_solve(**kwargs)

    assert min_eig > -tol, "Minimum eigenvalue not positive"
    assert np.abs((cost - cost_targ) / cost) < tol, "Cost does not agree with expected"


def test_p1_low_rank():
    low_rank_test(prob_file="test_prob_1.pkl", rank=2)


def test_p3_low_rank():
    low_rank_test(prob_file="test_prob_3.pkl", rank=2)


if __name__ == "__main__":
    test_p1_low_rank()
    test_p3_low_rank()
