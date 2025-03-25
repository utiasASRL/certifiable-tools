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
from cert_tools.rank_reduction import (
    get_low_rank_factor,
    rank_inflation,
    rank_inflation_lr,
    rank_inflation_pg,
    rank_inflation_sdp,
    rank_reduction,
)
from cert_tools.sdp_solvers import (
    adjust_Q,
    solve_lambda_cvxpy,
    solve_low_rank_sdp,
    solve_sdp_fusion,
    solve_sdp_homqcqp,
)

RANK_TOL = 1e-5


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


def sym_dim(n):
    return n * (n + 1) / 2


def run_test(test=4, load_solns=True, plot=False):
    # Load problems
    df_1 = load_dataset(fname="_examples/mw_slam_10pose_r1.pkl")
    prob_l = df_1.homQCQP[0]
    df_2 = load_dataset(fname="_examples/mw_slam_10pose_r0.pkl")
    prob_h = df_2.homQCQP[0]
    # Make the variable orderings consistent
    prob_h.get_asg(var_list=prob_l.var_list)

    # Get constraints
    C, constraints = get_cost_constraints(prob_h)
    C_l, constraints_l = get_cost_constraints(prob_l)

    # Run optimizations
    if load_solns:
        with open("saved_soln.pkl", "rb") as handle:
            data = pickle.load(handle)
            X_l = data["X_l"]
            X_h = data["X_h"]
            X_pr = data["X_pr"]
            constraints_pr = data["constraints_pr"]
    else:
        X_l, info_l, solve_time_l = solve_sdp_homqcqp(
            problem=prob_l, verbose=True, adjust=True
        )
        X_h, info_h, solve_time_h = solve_sdp_homqcqp(
            problem=prob_h, verbose=True, adjust=True
        )
        # Get rank-1 solution
        V_l, r = get_low_rank_factor(X_l, rank_tol=RANK_TOL)
        # Prune constraints to just get cost tightness
        first_redun = len(constraints) - 1
        constraints_pr = prune_constraints(
            C=C,
            Constraints=constraints_l,
            first_redun=first_redun,
            xhat=V_l,
            mult_cutoff=0.1,
        )
        # Resolve optimization problem with fewer constraints
        X_pr, info_pr = solve_sdp_fusion(
            Q=C, Constraints=constraints_pr, adjust=True, verbose=True
        )
        # Save data
        data = dict(
            X_l=X_l,
            X_h=X_h,
            X_pr=X_pr,
            info_pr=info_pr,
            constraints_pr=constraints_pr,
            info_l=info_l,
            info_h=info_h,
        )
        with open("saved_soln.pkl", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Get low rank factors
    V_l = get_low_rank_factor(X_l, rank_tol=RANK_TOL)
    V_h = get_low_rank_factor(X_h, rank_tol=RANK_TOL)  # Optimal cost
    p_opt = np.trace(C @ X_l)
    # Adjusted C
    C_adj, scale, offset = adjust_Q(C)

    # TESTS
    if test == 1:
        # Test 1: Check if range spaces overlap and reduce rank
        result = np.linalg.lstsq(V_h, V_l)
        residual = result[1]
        if residual > 1e-7:
            print("Low rank solution is not in the range space of high rank solution")
        else:
            print("Low rank solution is in the range space of high rank solution")

        # Check if the rank of the high rank solution can be reduced
        V_h_red = rank_reduction(constraints, X_h, rank_tol=RANK_TOL, verbose=True)
    elif test == 2:
        # Test 2: Check if the rank can be reduced if we use the redundant constraints as well
        V_h_red = rank_reduction(constraints_l, X_h, rank_tol=RANK_TOL, verbose=True)
    elif test == 3:
        # Test 3: Try to prune constraints so that we just get cost tightness and evaluate whether RRA works
        # Get optimality gap
        d_opt = np.trace(C @ X_pr)
        # Check if low rank solution is in the range space of this solution
        V_p, _ = get_low_rank_factor(X_pr, rank_tol=RANK_TOL)
        result = np.linalg.lstsq(V_p, V_l)
        residual = result[1]
        print(f"Range space check residual: {residual}")
        print(f"Reduced to {len(constraints_pr)}/{len(constraints_l)} constraints")
        print(f"Certified optimal cost:{p_opt}")
        print(f"Optimal cost with pruned constraints: {d_opt}")
        print("Applying rank reduction with pruned constraints...")
        V_pr = rank_reduction(
            constraints_pr, X_pr, rank_tol=RANK_TOL, null_tol=1e-5, verbose=True
        )
        # Check gap to redundant solution and constraint violation
        viol, cost_diff = check_feasibility(V_pr @ V_pr.T, p_opt, C, constraints)
        print(f"Max Violation: {np.max(np.abs(viol))}, Cost Delta: {cost_diff}")

        # Check similarity of solutions
        mag_l = np.linalg.norm(V_l)
        mag_pr = np.linalg.norm(V_pr)
        angle = np.arccos(V_l.T @ V_pr / mag_l / mag_pr)
        print(f"Mag of V_l: {mag_l}, Mag of V_pr {mag_pr}")
        print(f"Angle between vectors: {angle}")
        # Check if same certificate works
        H = data["info_pr"]["H"]
        print(f"Strong duality check at solution, V' @ H @ V = 0: {V_pr.T @ H@ V_pr}")
        print(f"Null space check, max(abs(H @ V)) = 0: {np.max(np.abs(H@ V_pr))}")
        if plot:
            # compare eigenvalues of certificates
            vals_l = np.linalg.eigvalsh(data["info_l"]["H"])
            vals_p = np.linalg.eigvalsh(H)
            plt.semilogy(vals_l, ".-", label="Rank Tight Certificate")
            plt.semilogy(vals_p, ".-", label="Cost Tight Certificate")
            plt.title("Dual/Certificate Eigenvalues")
            plt.legend()
            plt.grid(True)
        # Apply other rank reductions
        V_h = rank_reduction(
            constraints, X_h, rank_tol=RANK_TOL, null_tol=1e-5, verbose=False
        )

        # Print out costs
        print(f"Costs")
        print("=========")
        print(f"No Redundant: {np.trace(C@X_h)}")
        print(f"No Redundant (reduced): {np.trace(V_h.T @ C @ V_h)}")
        print(f"All Redundant: {np.trace(C@X_l)}")
        print(f"Cost Tight: {np.trace(C@X_pr)}")
        print(f"Cost Tight (reduced): {np.trace(V_pr.T @ C @ V_pr)}")

        # Plot eigenvalues for different cases
        if plot == True:
            eigvals_l = np.linalg.eigvalsh(X_l)
            eigvals_h = np.linalg.eigvalsh(X_h)
            eigvals_h_red = np.linalg.eigvalsh(V_h @ V_h.T)
            eigvals_pruned = np.linalg.eigvalsh(X_pr)
            eigvals_pruned_red = np.linalg.eigvalsh(V_pr @ V_pr.T)

            # Sort eigenvalues in descending order and take the largest 10
            largest_eigvals_l = np.sort(eigvals_l)[-10:][::-1]
            largest_eigvals_h = np.sort(eigvals_h)[-10:][::-1]
            largest_eigvals_h_red = np.sort(eigvals_h_red)[-10:][::-1]
            largest_eigvals_pruned = np.sort(eigvals_pruned)[-10:][::-1]
            largest_eigvals_pruned_red = np.sort(eigvals_pruned_red)[-10:][::-1]

            # Plot eigenvalues
            plt.figure(figsize=(10, 6))
            plt.semilogy(largest_eigvals_l, ".-", label="Tight")
            plt.semilogy(largest_eigvals_h, ".-", label="Not Tight")
            plt.semilogy(largest_eigvals_h_red, ".-", label="Not Tight (reduced)")
            plt.semilogy(largest_eigvals_pruned, ".-", label="Cost Tight")
            plt.semilogy(largest_eigvals_pruned_red, ".-", label="Cost Tight (reduced)")
            plt.xlabel("Index")
            plt.ylabel("Eigenvalue")
            plt.title("Largest 10 Eigenvalues")
            plt.legend()
            plt.grid(True)
            plt.show()
    elif test == 4:
        # Compare feasibility of RRA then round, versus just round.
        # Get rank reduction solution
        V_rr = rank_reduction(
            constraints, X_h, rank_tol=RANK_TOL, targ_rank=1, verbose=True
        )
        # Get truncated solution
        V_tr, rank = get_low_rank_factor(X_h, rank=1)
        # Get violations
        viol_rr, cost_diff_rr = check_feasibility(V_rr @ V_rr.T, 0, C, constraints)
        viol_tr, cost_diff_tr = check_feasibility(V_tr @ V_tr.T, p_opt, C, constraints)
        print("Cost difference convex lower bound")
        print(f"TR: {cost_diff_tr}, RR: {cost_diff_rr}")
        print("Maximum Constraint Violation")
        print(f"TR: {np.max(np.abs(viol_tr))}, RR: {np.max(np.abs(viol_rr))}")
    elif test == 5:
        # Check if we can reconstruct a certificate from complementary slackness of the cost tight solution

        # Get optimality gap
        d_opt = np.trace(C @ X_pr)
        # Low rank factor of of the cost tight solution
        V_p, r = get_low_rank_factor(X_pr, rank_tol=1e-6)
        # normalize the vectors
        # V_p = V_p / np.linalg.norm(V_p, axis=0, keepdims=True)
        # Check if we have a theoretically unique solution
        m = len(constraints_pr)
        n = V_p.shape[0]
        print(
            f"If the solution is unique then ({sym_dim(n-r)} <= {sym_dim(n)-m}) is true"
        )

        # Construct Constraint matrix
        A_bar = []
        for A, b in constraints_pr:
            A_bar.append(np.reshape(A @ V_p, (-1, 1), order="F"))
        A_bar = np.hstack(A_bar)
        c_bar = np.reshape(C_adj @ V_p, (-1, 1), order="F")
        # Check rank of constraint gradient
        u, s, v = np.linalg.svd(A_bar)
        print(f"Rank of constraint gradient matrix: {np.sum(s > s[0]*RANK_TOL)}")
        print(f"Number of constraints: {len(constraints_pr)}")
        # Solve for multipliers
        result = np.linalg.lstsq(A_bar, -c_bar)
        mults = result[0]
        residual = result[1]
        # Check that the residual is low
        print(f"Multiplier least squares solution residual: {np.abs(residual)}")
        # Construct certificate
        H = C_adj.toarray()
        for i, (A, b) in enumerate(constraints_pr):
            H += mults[i, 0] * A.toarray()
        vals = np.linalg.eigvalsh(H)
        # Print information
        print(
            f"Complementary Slackness (H@X) residual for rank-{r} solution: {np.trace(V_pr.T@ H @ V_pr)}"
        )
        print(
            f"Complementary Slackness (H@X) residual for rank-1 solution: {np.trace(H @ X_l)}"
        )
        print(f"Minimum certificate eigenvalue: {np.min(vals)}")
    elif test == 6:
        # Test 6 : Rank inflation via SDP
        X_max = rank_inflation_sdp(C, constraints_pr, X_l, verbose=True)
        s = np.linalg.eigvalsh(X_max)
        rank = np.sum(s > s[-1] * RANK_TOL)
        print(f"Rank of max rank solution: {rank}")
    elif test == 7:
        # Test 7 : Rank inflation via projected grad
        X_max = rank_inflation_pg(C, constraints_pr, X_l, verbose=True)
        s = np.linalg.eigvalsh(X_max)
        rank = np.sum(s > s[-1] * RANK_TOL)
        print(f"Rank of max rank solution: {rank}")
    elif test == 8:
        # Test 8 : Rank inflation via projected grad
        X_max = rank_inflation(C, constraints_pr, X_l, verbose=True)
        s = np.linalg.eigvalsh(X_max)
        rank = np.sum(s > s[-1] * RANK_TOL)
        print(f"Rank of max rank solution: {rank}")
    else:
        raise ValueError("Test unknown")


if __name__ == "__main__":
    run_test(test=8, load_solns=True, plot=True)
