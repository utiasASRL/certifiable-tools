import os
import pickle
import time

import matplotlib.pylab as plt
import numpy as np
from pandas import DataFrame, read_pickle
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
    print("Converting problem to HomQCQP")
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
    # Initialize
    problem.get_asg()
    # Remove any linearly independant constraints
    # bad_idx = problem.remove_dependent_constraints()
    # Clique decomposition
    problem.clique_decomposition(merge_function=problem.merge_cosmo)

    return problem


def process_problems(fname="_examples/mw_loc_3d_small.pkl"):
    # retieve data
    df = load_dataset(fname)
    # convert problems
    problems = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        problems.append(convert_to_qcqp(row))
    df["hom_qcqp"] = problems
    # save to new dataframe
    filename, ext = fname.split(".")
    filename += "_processed.pkl"
    with open(filename, "wb") as file:
        pickle.dump(df, file)


def test_problem(
    fname="_examples/mw_loc_3d_small.pkl",
    form="primal",
    id=8,
):
    problem, df = load_problem(fname=fname, id=id)

    problem.plot_asg(remove_vars=["h"], html="asg.html")
    problem.plot_ctree(html="ctree.html")

    # Solve standard SDP
    # X, info, solve_time = solve_sdp_homqcqp(problem, tol=1e-10, verbose=True)
    # x_full = X[:, 0]

    # Solve decomposed SDP
    methods = dict(objective="split", constraint="split")
    clq_list, info = solve_dsdp(
        problem, form=form, tol=1e-8, decomp_methods=methods, verbose=True
    )
    Y, ranks, factor_dict = problem.get_mr_completion(clq_list)

    # Check solution
    assert Y.shape[1] == 1, ValueError("Decomposed solution is not rank 1")
    x = []
    for varname in df.var_sizes.keys():
        x.append(factor_dict[varname].flatten())
    x = np.concatenate(x)
    if factor_dict["h"][0, 0] < 0:
        x = -x

    cost_decomp = x @ df.cost @ x
    cost_local = df.x_gt_init @ df.cost @ df.x_gt_init
    tol = 1e-3
    assert cost_decomp <= cost_local + tol, ValueError("Decomposed SDP has higher cost")

    if cost_local - cost_decomp > tol:
        print(f"Cost difference: {cost_local-cost_decomp}")
        print(f"Cost local: {cost_local}")
        print(f"Cost decomp: {cost_decomp}")
        # Check violations
        viol_local = []
        viol_decomp = []
        for A, b in df.constraints:
            viol_local.append(np.linalg.norm(df.x_gt_init.T @ A @ df.x_gt_init - b))
            viol_decomp.append(np.linalg.norm(x.T @ A @ x - b))
        print(f"Violations local: {np.linalg.norm(viol_local)}")
        print(f"Violations decomp: {np.linalg.norm(viol_decomp)}")
    else:
        print("Cost difference is within tolerance. checking solution")
        np.testing.assert_allclose(
            x, df.x_gt_init, atol=1e-6, err_msg="solution doesn't match"
        )


def run_speed_tests(dataset="mw_loc_3d", tol=1e-10):
    # retieve data
    df = load_dataset("_examples/" + dataset + ".pkl")
    # test problems
    results_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if "hom_qcqp" in row:
            problem = row["hom_qcqp"]
        else:
            problem = convert_to_qcqp(row)

        # Solve decomposed SDP
        clq_list, info = solve_dsdp(problem, tol=tol, verbose=True)
        Y, ranks, factor_dict = problem.get_mr_completion(clq_list)
        # Check solution
        assert Y.shape[1] == 1, ValueError("Decomposed solution is not rank 1")
        x = []
        for varname in row.var_sizes.keys():
            x.append(factor_dict[varname].flatten())
        x = np.concatenate(x)
        np.testing.assert_allclose(
            x, row.x_gt_init, atol=5e-4, err_msg="solution doesn't match"
        )
        # Store data
        info["x_sdp"] = x
        results_data.append(info)

    results = DataFrame(results_data)
    results.to_pickle("_results/" + dataset + "_results.pkl")


def plot_results(dataset="mw_loc_3d"):

    parameters = read_pickle("_examples/" + dataset + ".pkl")
    results = read_pickle("_results/" + dataset + "_results.pkl")

    plt.loglog(parameters["n params"], results["runtime"], ".-")
    plt.show()


if __name__ == "__main__":
    # test_problem(fname="_examples/mw_loc_3d_small.pkl")
    # test_problem(fname="_examples/mw_loc_3d.pkl", form="dual", id=4)
    # test_problem(fname="_examples/rangeonlyloc2d_no_const-vel_small.pkl")
    # test_problem(fname="_examples/rangeonlyloc2d_no_const-vel.pkl", id=27)

    # Process Problems
    # process_problems(fname="_examples/mw_loc_3d_small.pkl")
    # process_problems(fname="_examples/rangeonlyloc2d_no_const-vel_small.pkl")
    # process_problems(fname="_examples/rangeonlyloc2d_no_const-vel.pkl")
    # process_problems(fname="_examples/mw_loc_3d.pkl")

    # Run Speed test
    # run_speed_tests(dataset="mw_loc_3d", tol=1e-6)
    run_speed_tests(dataset="rangeonlyloc2d_no_const-vel", tol=1e-8)

    # Plot Results
    # plot_results()
