import itertools
import os

import numpy as np
from poly_matrix import PolyMatrix

from cert_tools import HomQCQP

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def generate_random_matrix(seed=0):
    np.random.seed(seed)
    dim_x = 2
    X = PolyMatrix()
    for i in range(1, 4):
        random_fill = np.random.rand(1 + 2 * dim_x, 1 + 2 * dim_x)
        random_fill += random_fill.T
        clique, __ = PolyMatrix.init_from_sparse(
            random_fill, {"h": 1, f"x_{i}": dim_x, f"x_{i+1}": dim_x}
        )
        clique.symmetric = True
        X += clique
    return X


def test_decompositions():
    def test_cost(cliques, C_gt):
        C = PolyMatrix()
        mat_decomp = problem.decompose_matrix(problem.C, method="split")
        for clique in cliques:
            C += mat_decomp[clique.index]
        np.testing.assert_allclose(C.get_matrix_dense(variables), C_gt)

    problem = HomQCQP()
    problem.C = generate_random_matrix()
    problem.As = []
    variables = problem.C.get_variables()
    C_gt = problem.C.get_matrix_dense(variables)

    # will create a clique decomposition that is not always the same
    problem.clique_decomposition()
    test_cost(problem.cliques, C_gt)

    clique_list = [
        {"h", "x_1", "x_2"},
        {"h", "x_2", "x_3"},
        {"h", "x_3", "x_4"},
    ]
    problem.clique_decomposition(clique_data=clique_list)
    test_cost(problem.cliques, C_gt)
    for var_list, clique in zip(clique_list, problem.cliques):
        assert set(clique.var_list).difference(var_list) == set()

    clique_data = {
        "cliques": [{"h", "x_1", "x_2"}, {"h", "x_2", "x_3"}, {"h", "x_3", "x_4"}],
        "separators": [{"h", "x_2"}, {"h", "x_3"}, {}],
        "parents": [
            1,
            2,
            2,
        ],  # parent of 0 is 1, parent of 1 is 2, parent of 2 is itself (it's the root)
    }
    problem.clique_decomposition(clique_data=clique_data)
    test_cost(problem.cliques, C_gt)
    for var_list, clique in zip(clique_list, problem.cliques):
        assert set(clique.var_list).difference(var_list) == set()


if __name__ == "__main__":
    test_decompositions()
    print("all tests passed.")
