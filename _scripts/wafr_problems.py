import os
import pickle
import time

import matplotlib.pylab as plt
import numpy as np
from poly_matrix import PolyMatrix
from tqdm import tqdm

from cert_tools.hom_qcqp import HomQCQP
from cert_tools.sdp_solvers import solve_sdp_homqcqp
from cert_tools.sparse_solvers import solve_clarabel, solve_dsdp

np.set_printoptions(precision=2)

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def load_dataset(fname="_examples/mw_loc_3d_small.pkl"):
    with open(fname, "rb") as file:
        df = pickle.load(file)
    return df


def load_problem(fname="_examples/mw_loc_3d_small.pkl", id=8):
    # retieve data
    df = load_dataset(fname)
    if id is not None:
        df = df.loc[id]
    # convert to Homogeneous QCQP
    problem = convert_to_qcqp(df)

    return problem, df


def convert_to_qcqp(df):
    """Convert data from dataframe into homogenized QCQP form

    Args:
        df (_type_): dataframe with one element
    """
    # convert to Homogeneous QCQP
    problem = HomQCQP(homog_var="h")
    var_sizes = df["var_sizes"]
    problem.C, _ = PolyMatrix.init_from_sparse(
        df["cost"], var_dict=var_sizes, symmetric=True
    )
    problem.As = []
    for A, b in df["constraints"]:
        if b == 0.0:
            pmat, _ = PolyMatrix.init_from_sparse(A, var_dict=var_sizes, symmetric=True)
            problem.As.append(pmat)
    # Initialize and get clique decomposition
    problem.get_asg()
    problem.clique_decomposition(merge_function=problem.merge_cosmo)

    return problem


def process_problems(fname="_examples/mw_loc_3d_small.pkl"):
    # retieve data
    df = load_dataset(fname)
    # convert problems
    problems = []
    for index, row in tqdm(df.iterrows()):
        problems.append(convert_to_qcqp(row))
    df["hom_qcqp"] = problems
    # save to new dataframe
    filename, ext = fname.split(".")
    filename += "_processed.pkl"
    with open(filename, "wb") as file:
        pickle.dump(df, file)


def test_problem(
    fname="_examples/mw_loc_3d_small.pkl",
    id=8,
):
    problem, df = load_problem(fname=fname, id=id)

    problem.plot_asg(remove_vars=["h"], html="asg.html")
    problem.plot_ctree(html="ctree.html")

    # Solve standard SDP
    # X, info, solve_time = solve_sdp_homqcqp(problem, tol=1e-10, verbose=True)

    # Solve decomposed SDP
    clq_list, info = solve_dsdp(problem, tol=1e-10, verbose=True)
    Y, ranks, factor_dict = problem.get_mr_completion(clq_list)

    # Check solution
    assert Y.shape[1] == 1, ValueError("Decomposed solution is not rank 1")
    x = []
    for varname in df.var_sizes.keys():
        x.append(factor_dict[varname].flatten())
    x = np.concatenate(x)
    np.testing.assert_allclose(
        x, df.x_gt_init, atol=5e-5, err_msg="solution doesn't match"
    )


if __name__ == "__main__":
    # test_problem(fname="_examples/mw_loc_3d_small.pkl")
    # test_problem(fname="_examples/mw_loc_3d.pkl")
    # test_problem(fname="_examples/rangeonlyloc2d_no_const-vel_small.pkl")
    test_problem(fname="_examples/rangeonlyloc2d_no_const-vel.pkl", id=27)
    # process_problems(fname="_examples/mw_loc_3d_small.pkl")
    # process_problems(fname="_examples/rangeonlyloc2d_no_const-vel_small.pkl")
