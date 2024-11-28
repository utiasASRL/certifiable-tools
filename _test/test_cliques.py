import os

import numpy as np
from cert_tools import HomQCQP
from cert_tools.test_tools import constraints_test, cost_test, get_chain_rot_prob

from poly_matrix import PolyMatrix

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def generate_random_matrix(seed=0):
    """Creates random block-tridiagonal arrowhead matrix"""
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


def test_symmetric():
    """Making sure that setting symmetric to True after the fact doesn't break anything.
    This test could probably go inside poly_matrix."""
    X_poly = generate_random_matrix()
    X_sparse = X_poly.get_matrix()

    X_poly_test, __ = PolyMatrix.init_from_sparse(X_sparse, X_poly.variable_dict_i)
    X_poly_test.symmetric = True
    for key_i in X_poly.variable_dict_i:
        for key_j in X_poly.adjacency_j[key_i]:
            np.testing.assert_allclose(X_poly_test[key_i, key_j], X_poly[key_i, key_j])

    X_poly_test, __ = PolyMatrix.init_from_sparse(X_sparse, X_poly.variable_dict_i)
    for key_i in X_poly.variable_dict_i:
        for key_j in X_poly.adjacency_j[key_i]:
            np.testing.assert_allclose(X_poly_test[key_i, key_j], X_poly[key_i, key_j])


def test_constraint_decomposition():

    problem = get_chain_rot_prob()
    problem.clique_decomposition()
    constraints_test(problem)


def test_cost_decomposition():

    problem = HomQCQP()
    problem.C = generate_random_matrix()
    problem.As = []

    # will create a clique decomposition that is not always the same
    problem.clique_decomposition()
    cost_test(problem)

    clique_list = [
        {"h", "x_1", "x_2"},
        {"h", "x_2", "x_3"},
        {"h", "x_3", "x_4"},
    ]
    problem.clique_decomposition(clique_data=clique_list)
    cost_test(problem)
    for var_list, clique in zip(clique_list, problem.cliques):
        assert set(clique.var_list).difference(var_list) == set()

    # parent of 0 is 1, parent of 1 is 2, parent of 2 is itself (it's the root)
    clique_data = {
        "cliques": [{"h", "x_1", "x_2"}, {"h", "x_2", "x_3"}, {"h", "x_3", "x_4"}],
        "separators": [{"h", "x_2"}, {"h", "x_3"}, {}],
        "parents": [1, 2, 2],
    }
    problem.clique_decomposition(clique_data=clique_data)
    cost_test(problem)
    for var_list, clique in zip(clique_list, problem.cliques):
        assert set(clique.var_list).difference(var_list) == set()


if __name__ == "__main__":
    test_symmetric()
    test_constraint_decomposition()
    test_cost_decomposition()
    print("all tests passed.")
