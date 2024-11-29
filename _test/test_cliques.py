import os

import numpy as np
from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.test_tools import (constraints_test, cost_test,
                                   get_chain_rot_prob)

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def generate_random_matrix(seed=0):
    """Creates random block-tridiagonal arrowhead matrix"""
    np.random.seed(seed)
    dim_x = 2
    X = PolyMatrix()
    n_vars = 4
    var_sizes = {"h": 1}
    var_sizes.update({f"x_{i}": dim_x for i in range(1, n_vars)})
    for i in range(1, n_vars):
        random_fill = np.random.rand(1 + 2 * dim_x, 1 + 2 * dim_x)
        random_fill += random_fill.T
        clique, __ = PolyMatrix.init_from_sparse(
            random_fill, {"h": 1, f"x_{i}": dim_x, f"x_{i+1}": dim_x}
        )
        clique.symmetric = True
        X += clique
    return X, var_sizes


def test_symmetric():
    """Making sure that setting symmetric to True after the fact doesn't break anything.
    This test could probably go inside poly_matrix."""
    X_poly, var_sizes = generate_random_matrix()
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
    problem.C, var_sizes = generate_random_matrix()
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


def get_chain_clique_data(var_sizes, fixed=["h"], variable=["x_", "z_"]):
    clique_data = []
    indices = [
        int(v.split(variable[0])[1].strip("_")) for v in var_sizes if variable[0] in v
    ]
    # debug start
    if len(variable) > 1:
        for v in variable:
            for i in indices:
                assert f"{v}{i}" in var_sizes
    # debug end
    for i, j in zip(indices[:-1], indices[1:]):
        clique_data.append(
            set(fixed + [f"{v}{i}" for v in variable] + [f"{v}{j}" for v in variable])
        )
    return clique_data


def test_fixed_decomposition():
    """Example of how to do a clique decomposition keeping the order of variables within 
    each clique."""
    problem = HomQCQP()
    problem.C, var_sizes = generate_random_matrix()
    problem.As = []

    problem.get_asg(var_list=var_sizes)

    clique_data = get_chain_clique_data(var_sizes, fixed=["h"], variable=["x_"])
    problem.clique_decomposition(clique_data=clique_data)

    for c, vars in zip(problem.cliques, clique_data):
        assert len(set(c.var_list) - vars) == 0


if __name__ == "__main__":
    test_fixed_decomposition()
    test_symmetric()
    test_constraint_decomposition()
    test_cost_decomposition()
    print("all tests passed.")
    print("all tests passed.")
