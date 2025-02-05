import pickle

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
import sdprlayers.utils.ess_mat_utils as utils
import torch
from poly_matrix import PolyMatrix
from sdprlayers.utils.camera_model import CameraModel
from sdprlayers.utils.lie_algebra import so3_exp, so3_log, so3_wedge

from cert_tools import HomQCQP
from cert_tools.rank_reduction import get_low_rank_factor, rank_reduction
from cert_tools.sdp_solvers import (
    solve_lambda_cvxpy,
    solve_low_rank_sdp,
    solve_sdp_fusion,
    solve_sdp_homqcqp,
)


def load_dataset(fname="_examples/mw_slam_10pose_r1.pkl"):
    with open(fname, "rb") as file:
        df = pickle.load(file)
    return df


# get cost and constraints
def get_cost_constraints(homQCQP, var_sizes=None):
    if var_sizes is None:
        var_sizes = homQCQP.var_sizes
    C = homQCQP.C.get_matrix(variables=var_sizes)
    As = homQCQP.As
    constraints = [(A.get_matrix(variables=var_sizes), 0.0) for A in As]
    # Add homogenizing constraint
    h = homQCQP.h
    A_0 = PolyMatrix()
    A_0[h, h] = 1.0
    constraints.append((A_0.get_matrix(variables=var_sizes), 1.0))
    return C, constraints


def check_feasibility(X, p_opt, C, constraints):
    """Check feasibility of the solution."""
    # Check the constraints
    viol = []
    for A, b in constraints:
        viol.append(np.trace(A @ X) - b)
    # Check the cost
    cost_diff = np.abs(np.trace(C @ X) - p_opt)

    return viol, cost_diff


def prune_constraints(C, Constraints, first_redun, xhat, mult_cutoff=10):
    X, mults, H = solve_lambda_cvxpy(
        Q=C,
        Constraints=Constraints,
        xhat=xhat,
        single_affine=True,
        fixed_epsilon=1e-8,
        adjust=False,
        verbose=True,
        tol=1e-8,
        force_first=first_redun,
    )
    constraints_out = []
    for i in range(len(mults) - 1):  # Note remove homog constraint
        if np.abs(mults[i]) > mult_cutoff or i < first_redun:
            constraints_out.append(Constraints[i])
    # Add back homogenizing constraint
    constraints_out.append(Constraints[-1])
    return constraints_out


def run_test(test=4, load_solns=True, plot=False):
    # Load problems
    df_1 = load_dataset(fname="_examples/mw_slam_10pose_r1.pkl")
    prob_r = df_1.homQCQP[0]
    df_2 = load_dataset(fname="_examples/mw_slam_10pose_r0.pkl")
    prob_nr = df_2.homQCQP[0]
    # Make the variable orderings consistent
    prob_nr.get_asg(var_list=prob_r.var_list)

    # Run optimizations
    if load_solns:
        with open("saved_soln.pkl", "rb") as handle:
            data = pickle.load(handle)
            X_r = data["X_r"]
            X_nr = data["X_nr"]
    else:
        X_r, info_r, solve_time_r = solve_sdp_homqcqp(
            problem=prob_r, verbose=True, adjust=True
        )
        X_nr, info_nr, solve_time_nr = solve_sdp_homqcqp(
            problem=prob_nr, verbose=True, adjust=True
        )
        data = dict(X_r=X_r, X_nr=X_nr, info_r=info_r, info_nr=info_nr)
        with open("saved_soln.pkl", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Get low rank factors
    V_r, r = get_low_rank_factor(X_r, rank_tol=1e-5)
    V_nr, r = get_low_rank_factor(X_nr, rank_tol=1e-5)

    # Get constraints
    C, constraints = get_cost_constraints(prob_nr)
    C_r, constraints_r = get_cost_constraints(prob_r)
    # Optimal cost
    p_opt = np.trace(C @ X_r)

    if test == 1:
        # Test 1: Check if range spaces overlap
        result = np.linalg.lstsq(V_nr, V_r)
        residual = result[1]
        if residual > 1e-7:
            print("Low rank solution is not in the range space of high rank solution")
        else:
            print("Low rank solution is in the range space of high rank solution")

    elif test == 2:
        # Test 2: Check if the rank of the high rank solution can be reduced
        V_nr_red = rank_reduction(
            constraints, X_nr, rank_tol=1e-5, null_method="svd", verbose=True
        )
    elif test == 3:
        # Test 3: Check if the rank can be reduced if we use the redundant constraints as well
        V_nr_red = rank_reduction(
            constraints_r, X_nr, rank_tol=1e-5, null_method="svd", verbose=True
        )
    elif test == 4:
        # Test 4: Try to prune constraints so that we just get cost tightness.
        # Get new solution with pruned constraints. Reduce rank.

        # Prune Constraints
        first_redun = len(constraints) - 1
        constraints_pruned = prune_constraints(
            C=C,
            Constraints=constraints_r,
            first_redun=first_redun,
            xhat=V_r,
            mult_cutoff=0.1,
        )
        # Resolve optimization problem
        X_pruned, info = solve_sdp_fusion(
            Q=C, Constraints=constraints_pruned, adjust=True, verbose=True
        )
        # Get optimality gap
        d_opt = np.trace(C @ X_pruned)
        gap = (p_opt - d_opt) / d_opt

        print(
            f"Gap to optimal with {len(constraints_pruned)}/{len(constraints_r)} constraints: {gap}"
        )
        print("Applying rank reduction with pruned constraints...")
        V_pr = rank_reduction(
            constraints_pruned, X_pruned, rank_tol=1e-5, null_tol=1e-5, verbose=True
        )
        # Check gap to redundant solution and constraint violation
        viol, cost_diff = check_feasibility(V_pr @ V_pr.T, p_opt, C, constraints)
        print(f"Max Violation: {np.max(np.abs(viol))}, Cost Delta: {cost_diff}")
        d_opt = V_pr.T @ C @ V_pr
        gap = (p_opt - d_opt) / d_opt
        print(f"Gap to optimal with Reduced-Rank: {gap[0,0]}")

        # Check if same certificate works
        H = info["H"]
        print(f"Strong duality check, V' @ H @ V = 0: {V_pr.T @ H@ V_pr}")
        print(f"Null space check, max(abs(H @ V)) = 0: {np.max(np.abs(H@ V_pr))}")

        # Apply other rank reductions
        V_nr = rank_reduction(
            constraints, X_nr, rank_tol=1e-5, null_tol=1e-5, verbose=False
        )

        # Print out costs
        print(f"Costs")
        print("=========")
        print(f"No Redundant: {np.trace(C@X_nr)}")
        print(f"No Redundant (reduced): {np.trace(V_nr.T @ C @ V_nr)}")
        print(f"All Redundant: {np.trace(C@X_r)}")
        print(f"Cost Tight: {np.trace(C@X_pruned)}")
        print(f"Cost Tight (reduced): {np.trace(V_pr.T @ C @ V_pr)}")

        # Plot eigenvalues for different cases
        if plot == True:
            eigvals_r = np.linalg.eigvalsh(X_r)
            eigvals_nr = np.linalg.eigvalsh(X_nr)
            eigvals_nr_red = np.linalg.eigvalsh(V_nr @ V_nr.T)
            eigvals_pruned = np.linalg.eigvalsh(X_pruned)
            eigvals_pruned_red = np.linalg.eigvalsh(V_pr @ V_pr.T)

            # Sort eigenvalues in descending order and take the largest 10
            largest_eigvals_r = np.sort(eigvals_r)[-10:][::-1]
            largest_eigvals_nr = np.sort(eigvals_nr)[-10:][::-1]
            largest_eigvals_nr_red = np.sort(eigvals_nr_red)[-10:][::-1]
            largest_eigvals_pruned = np.sort(eigvals_pruned)[-10:][::-1]
            largest_eigvals_pruned_red = np.sort(eigvals_pruned_red)[-10:][::-1]

            # Plot eigenvalues
            plt.figure(figsize=(10, 6))
            plt.semilogy(largest_eigvals_r, ".-", label="Tight")
            plt.semilogy(largest_eigvals_nr, ".-", label="Not Tight")
            plt.semilogy(largest_eigvals_nr_red, ".-", label="Not Tight (reduced)")
            plt.semilogy(largest_eigvals_pruned, ".-", label="Cost Tight")
            plt.semilogy(largest_eigvals_pruned_red, ".-", label="Cost Tight (reduced)")
            plt.xlabel("Index")
            plt.ylabel("Eigenvalue")
            plt.title("Largest 10 Eigenvalues")
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    run_test(test=4, load_solns=True)
