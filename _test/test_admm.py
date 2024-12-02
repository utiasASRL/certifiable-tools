"""
ADMM test problem:

       [1-1 0 0 0]            [4-2 0 0 0]            [9-3 0 0 0]        
       [0 1 0 0 0]            [0 1 0 0 0]            [0 1 0 0 0]       
min  < [0 0 0 0 0], X_1 > + < [0 0 0 0 0], X_2 > + < [0 0 0 0 0], X_3 >
       [0 0 0 0 0]            [0 0 0 0 0]            [0 0 0 1 0]        
       [0 0 0 0 0]            [0 0 0 0 0]            [0 0 0 0 0]       

        homogeneous constraints:
           [1 0 0 0 0]
           [0 0 0 0 0]
    s.t. < [0 0 0 0 0], X_i > = 1   i = 1, 2, 3
           [0 0 0 0 0] 
           [0 0 0 0 0]

        primary constraints per clique: 
           [0 0 0 0 0]
           [0 1 0 0 0]
         < [0 0-1 0 0], X_i > = 0   i = 1, 2, 3
           [0 0 0 0 0] 
           [0 0 0 0 0]

        consensus constraints:
           [0 0 0 0 0]            [0 0 0 0 0]              
           [0 0 0 0 0]            [0-1 0 0 0]              
         < [0 0 0 0 0], X_1 > + < [0 0 0 0 0], X_2 > = 0   etc. 
           [0 0 0 1 0]            [0 0 0 0 0]              
           [0 0 0 0 0]            [0 0 0 0 0]              

Solution is:
 h  x_0   x_1   x_2   x_3 
[1  1  1  2  2  3  3  0  0]
"""

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from cert_tools.admm_clique import ADMMClique
from cert_tools.admm_solvers import solve_alternating
from cert_tools.hom_qcqp import HomQCQP
from cert_tools.linalg_tools import rank_project, svec
from cert_tools.sdp_solvers import solve_sdp
from cert_tools.test_tools import get_chain_rot_prob


def create_admm_test_problem():
    """
           [1-1 0 0 0]            [4-2 0 0 0]            [9-3 0 0 0]
           [0 1 0 0 0]            [0 1 0 0 0]            [0 1 0 0 0]
    min  < [0 0 0 0 0], X_1 > + < [0 0 0 0 0], X_2 > + < [0 0 0 0 0], X_3 >
           [0 0 0 0 0]            [0 0 0 0 0]            [0 0 0 1 0]
           [0 0 0 0 0]            [0 0 0 0 0]            [0 0 0 0 0]
    """
    Q = PolyMatrix()
    # first block
    Q["h", "h"] = 1.0
    Q["h", "x_0"] = np.array([[-1.0, 0]])
    Q["x_0", "x_0"] = np.array([[1.0, 0.0], [0.0, 0.0]])
    # second block
    Q["h", "h"] += 4.0
    Q["h", "x_1"] = np.array([[-2.0, 0]])
    Q["x_1", "x_1"] = np.array([[1.0, 0.0], [0.0, 0.0]])
    # third block
    Q["h", "h"] += 9.0
    Q["h", "x_2"] = np.array([[-3.0, 0]])
    Q["x_2", "x_2"] = np.array([[1.0, 0.0], [0.0, 0.0]])
    Q["x_3", "x_3"] = np.array([[1.0, 0.0], [0.0, 0.0]])

    # constraints
    """
        primary constraints per clique: 
           [0 0 0 0 0]
           [0 1 0 0 0]
         < [0 0-1 0 0], X_i > = 0   i = 1, 2, 3
           [0 0 0 0 0] 
           [0 0 0 0 0]
    """
    A_1a = PolyMatrix()
    A_1a["x_0", "x_0"] = np.array([[1.0, 0.0], [0.0, -1.0]])
    A_1b = PolyMatrix()
    A_1b["h", "x_0"] = np.array([[1.0, -1.0]])
    A_2a = PolyMatrix()
    A_2a["x_1", "x_1"] = np.array([[1.0, 0.0], [0.0, -1.0]])
    A_2b = PolyMatrix()
    A_2b["h", "x_1"] = np.array([[1.0, -1.0]])
    A_3a = PolyMatrix()
    A_3a["x_2", "x_2"] = np.array([[1.0, 0.0], [0.0, -1.0]])
    A_3b = PolyMatrix()
    A_3b["h", "x_2"] = np.array([[1.0, -1.0]])
    A_4a = PolyMatrix()
    A_4a["x_3", "x_3"] = np.array([[1.0, 0.0], [0.0, -1.0]])
    A_4b = PolyMatrix()
    A_4b["h", "x_3"] = np.array([[1.0, -1.0]])

    problem = HomQCQP()
    problem.C = Q
    problem.As = [A_1a, A_2a, A_3a, A_4a, A_1b, A_2b, A_3b, A_4b]
    problem.var_sizes = {"h": 1, "x_0": 2, "x_1": 2, "x_2": 2, "x_3": 2}
    return problem


def test_consistency():
    problem = create_admm_test_problem()
    admm_cliques = ADMMClique.create_admm_cliques_from_problem(problem, variable=["x_"])

    Q, Constraints = problem.get_problem_matrices()
    X, *_ = solve_sdp(Q, Constraints)
    X_poly, __ = PolyMatrix.init_from_sparse(X, var_dict=problem.var_sizes)

    for clique in admm_cliques:
        clique.X = X_poly.get_matrix_dense(clique.var_size)

    # check that vectorized consistency constraints hold everywhere.
    for clique in admm_cliques:
        g = np.vstack(
            [Gi @ admm_cliques[vi].X.reshape(-1, 1) for vi, Gi in clique.G_dict.items()]
        )
        np.testing.assert_allclose(
            clique.F @ clique.X.reshape(-1, 1),
            -g,
        )
    print("consistency tests passed")


def test_problem():

    problem = create_admm_test_problem()
    Q, Constraints = problem.get_problem_matrices()
    X, *_ = solve_sdp(Q, Constraints)
    print(X)
    plt.matshow(X)
    x, info_rank = rank_project(X)
    np.testing.assert_allclose(
        x.flatten(), [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0], atol=1e-5
    )
    print("done")


def plot_admm():
    problem = create_admm_test_problem()

    clique_data = [{"h", f"x_{i}", f"x_{i+1}"} for i in range(3)]
    problem.clique_decomposition(clique_data=clique_data)

    # C_dict = problem.decompose_matrix(problem.C, method="greedy-cover")
    # C_dict = problem.decompose_matrix(problem.C, method="first")
    C_dict = problem.decompose_matrix(problem.C, method="split")
    fig, axs = plt.subplots(1, len(C_dict), squeeze=False)
    axs = {i: ax for i, ax in zip(C_dict, axs[0, :])}
    for i, C in C_dict.items():
        C.matshow(ax=axs[i])
        axs[i].set_title(f"clique {i}")
    # plt.show(block=False)

    A_dicts = []
    for A in problem.As:
        A_dict = problem.decompose_matrix(A, method="first")
        fig, axs = plt.subplots(1, len(A_dict), squeeze=False)
        axs = {i: ax for i, ax in zip(A_dict, axs[0, :])}
        for i, A in A_dict.items():
            A.matshow(ax=axs[i])
            axs[i].set_title(f"clique {i}")
        A_dicts.append(A_dict)
    plt.close("all")

    eq_list = problem.get_consistency_constraints()
    assert len(eq_list) == 2 * 6
    counter = {(1, 0): 0, (2, 1): 0}
    plots = {(1, 0): plt.subplots(2, 6)[1], (2, 1): plt.subplots(2, 6)[1]}
    for k, l, Ak, Al in eq_list:
        # Ak.matshow(ax=plots[(k, l)][0, counter[(k, l)]])
        # Al.matshow(ax=plots[(k, l)][1, counter[(k, l)]])
        plots[(k, l)][0, counter[(k, l)]].matshow(Ak.toarray())
        plots[(k, l)][1, counter[(k, l)]].matshow(Al.toarray())
        counter[(k, l)] += 1
    return


def test_admm():
    problem = create_admm_test_problem()

    Q, Constraints = problem.get_problem_matrices()
    X, info_SDP = solve_sdp(Q, Constraints)
    print("cost SDP", info_SDP["cost"])

    admm_cliques = ADMMClique.create_admm_cliques_from_problem(problem, variable=["x_"])
    solution, info = solve_alternating(admm_cliques, verbose=True, maxiter=10)
    print("cost ADMM", info["cost"])


def notest_fusion():
    from cert_tools.fusion_tools import mat_fusion, svec_fusion

    problem = create_admm_test_problem()
    Q = problem.C.get_matrix(problem.var_sizes)

    Q_fusion = mat_fusion(Q)
    q_fusion = svec_fusion(Q_fusion)

    q = Q.toarray()[np.triu_indices[Q.shape[0]]]
    np.testing.assert_allclose(q, q_fusion)


if __name__ == "__main__":
    # plot_admm()
    # test_consistency()
    test_admm()
    # test_problem()
