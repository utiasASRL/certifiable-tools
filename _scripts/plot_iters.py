import os, sys
import pickle

import matplotlib.pyplot as plt

from matplotlib import cm

import numpy as np
import scipy.sparse as sp
import scipy.spatial as spatial

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")
sys.path.append(root_dir)
from cert_tools.eopt_solvers import *


def load_data(prob_file="test_prob_1.pkl"):
    try:
        with open(os.path.join(root_dir, "_examples", prob_file), "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Skipping {prob_file} cause file not found.")
        return None


def run_eopt(data, opts=opts_sbm_dflt, global_min=True, method="cuts"):
    # Get global solution
    if "x_cand" in data:
        x_cand = data["x_cand"]
    else:
        u, s, v = np.linalg.svd(data["X"])
        x_cand = u[:, [0]] * np.sqrt(s[0])
    # Run optimizer
    opts.update(dict(use_null=False))
    Q = data["Q"].copy()
    x, output = solve_eopt(
        Q=Q, Constraints=data["Constraints"], x_cand=x_cand, opts=opts, method=method
    )

    return output


def process_data(data, inds=[0, 1], method="sbm"):
    """Preprocess the data down to two multipliers"""
    # create vectorized constraints
    A_vec_list = []
    for i in range(len(data["Constraints"])):
        A, b = data["Constraints"][i]
        A_vec_list += [A.reshape((-1, 1), order="F")]
    A_vec = sp.hstack(A_vec_list)

    # Run eigenvalue optimization
    mults, info = solve_eopt_cuts(
        Q=data["Q"],
        A_vec=A_vec,
        opts=opts_cut_dflt,
        xinit=np.array([[0.0, 0.0]]).T,
        kwargs_eig=dict(method="direct"),
    )

    Q_bar = data["Q"]
    if len(mults) > 2:
        # Reduce the set of multipliers
        mults_red = []
        A_vec_list = []
        for i in range(len(data["Constraints"])):
            A, b = data["Constraints"][i]
            if i in inds:
                A_vec_list += [A.reshape((-1, 1), order="F")]
                mults_red += [mults[i]]
            else:
                Q_bar += mults[i] * A
        A_vec = sp.hstack(A_vec_list)
    else:
        mults_red = mults

    data.update(dict(Q_bar=Q_bar, A_vec=A_vec, mults=mults_red))

    return data


def plot_obj(data, rng=5):
    Q_bar = data["Q_bar"]
    A_vec = data["A_vec"]
    mults = data["mults"]
    # PLOT OBJECTIVE
    # Construct ranges
    n_pts = 50
    l1 = mults[0] + np.linspace(-rng, rng, n_pts)
    l2 = mults[1] + np.linspace(-rng, rng, n_pts)
    f = np.zeros((len(l1), len(l2)))
    for i in range(len(l1)):
        for j in range(len(l2)):
            # Get Certificate
            H = get_cert_mat(Q_bar, A_vec, [l1[i], l2[j]])
            # Get min eig
            grad_info = get_grad_info(H, A_vec, method="direct")
            f[i, j] = grad_info["min_eig"]

    # Plot Minimum Eigenvalues
    L1, L2 = np.meshgrid(l1, l2)
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    surf = ax.plot_surface(L1, L2, f.T, label="min eig", alpha=0.75, cmap=cm.YlGn)
    ax.view_init(elev=45, azim=-90, roll=0)
    ax.set_aspect("equal")
    return fig, ax


def plot_iters_sbm(data, x_init, fig, ax):
    # RUN OPTIMIZATION
    # Run Actual optimization with two multipliers
    solver = solve_eopt_sbm
    opts = opts_sbm_dflt
    alphas, info = solver(
        Q=data["Q_bar"],
        A_vec=data["A_vec"],
        A_eq=None,
        b_eq=None,
        xinit=x_init,
        opts=opts,
        kwargs_eig=dict(method="direct"),
    )
    print("STATUS:\t" + info["status"])
    # PLOT ITERATIONS
    x_vals = info["iter_info"]["x"].values
    if x_vals[0].shape[0] == 1:
        x_vals = np.vstack(x_vals)
    else:
        x_vals = np.hstack(x_vals).T
    m_eig = info["iter_info"]["min_eig_curr"].values[:, None]
    # Select colour based on null step
    delta = info["iter_info"]["delta_norm"]
    ind = delta > 1e-12
    null = delta < 1e-12
    iter_str = [str(i) for i in info["iter_info"]["n_iter"].values]
    ax.scatter3D(x_vals[ind, 0], x_vals[ind, 1], m_eig[ind, 0], marker="o", c="r")
    ax.scatter3D(
        x_vals[null, 0], x_vals[null, 1], m_eig[null, 0], marker="o", c="b", alpha=0.3
    )
    ax.plot3D(x_vals[:, 0], x_vals[:, 1], m_eig[:, 0], c="b", alpha=0.3)
    ax.plot3D(x_vals[ind, 0], x_vals[ind, 1], m_eig[ind, 0], c="r")
    for i in range(x_vals.shape[0]):
        if ind[i]:
            ax.text(x_vals[i, 0], x_vals[i, 1], m_eig[i, 0] + 0.1, iter_str[i])

    return fig, ax


def plot_iters(data, x_init, fig, ax, method="sbm"):
    # RUN OPTIMIZATION
    # Run Actual optimization with two multipliers
    if method == "cuts":
        solver = solve_eopt_cuts
        opts = opts_cut_dflt
    elif method == "sub":
        solver = solve_eopt_sub
        opts = opts_sub_dflt
    elif method == "sbm":
        solver = solve_eopt_sbm
        opts = opts_sbm_dflt
    else:
        raise ValueError(f"Unknown method {method} in solve_eopt")
    alphas, info = solver(
        Q=data["Q_bar"],
        A_vec=data["A_vec"],
        A_eq=None,
        b_eq=None,
        xinit=x_init,
        opts=opts,
        kwargs_eig=dict(method="direct"),
    )
    print("STATUS:\t" + info["status"])
    # PLOT ITERATIONS
    x_vals = info["iter_info"]["x"].values
    if len(x_vals[0].shape) == 1 or x_vals[0].shape[0] == 1:
        x_vals = np.vstack(x_vals)
    else:
        x_vals = np.hstack(x_vals).T
    m_eig = info["iter_info"]["min_eig_curr"].values[:, None]
    ax.scatter3D(x_vals[:, 0], x_vals[:, 1], m_eig[:, 0], marker="o", c="r")
    ax.plot3D(x_vals[:, 0], x_vals[:, 1], m_eig[:, 0], c="r")

    return fig, ax


def run_test_prob(prob_file="test_prob_8G.pkl"):
    # Get data
    data = load_data(prob_file=prob_file)
    data = tent_obj()
    # process
    data = process_data(data, inds=[0, 1])
    # plot objective
    fig, ax = plot_obj(data=data, rng=5)

    # SPECTRAL BUNDLE
    # plot iterations
    x_init = np.array([[0.0, -5.0]]).T
    fig, ax = plot_iters_sbm(data, x_init, fig, ax)
    ax.set_title("Spectral Bundle Method")

    # CUT PLANE
    # plot objective
    fig, ax = plot_obj(data=data)
    # plot iterations
    fig, ax = plot_iters(data, x_init, fig, ax, method="cuts")
    ax.set_title("Cut Plane")

    # SUBGRAD
    # plot objective
    fig, ax = plot_obj(data=data)
    # plot iterations
    fig, ax = plot_iters(data, x_init, fig, ax, method="sub")
    ax.set_title("Subgrad")

    plt.show()


def gen_hsphere_cloud(r=1, n_pts=10):
    """Generate points on the top hemisphere of a circle"""
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0 : pi * 3 / 8 : n_pts * 1j, 0.0 : 2.0 * pi : n_pts * 1j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi) - 0.98 * r

    pts = np.hstack([x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))])

    return pts


def get_hsphere_obj(r=1):
    # Generate hemisphere points
    pts = gen_hsphere_cloud(r=r)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.show()
    # Get convex hull of hsphere points
    hull = spatial.ConvexHull(pts)
    # get normals and offsets
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
    # Remove down facing normals
    ind = normals[:, 2] > 0.0
    normals = normals[ind, :]
    offsets = offsets[ind]
    # Define Matrices
    q = -offsets / normals[:, 2]
    B = -normals[:, :2] / normals[:, [2]]
    # Q = sp.diags(q)
    Q = sp.diags([q, 0.3 * np.ones(len(q) - 1)], [0, 1])
    A_1 = sp.diags(B[:, 0])
    A_2 = sp.diags(B[:, 1])
    Constraints = [(A_1, 0.0), (A_2, 0.0)]
    x = np.array([[0, 0]])
    X = x.T @ x
    data = dict(Q=Q, Constraints=Constraints, X=X)

    return data


def run_sphere_prob():
    # Get data
    # data = tent_obj_2()
    r = 10
    data = get_hsphere_obj(r=r)
    # process
    data = process_data(data, inds=[0, 1])
    # plot objective
    fig, ax = plot_obj(data=data, rng=r)

    # SPECTRAL BUNDLE
    # plot iterations
    x_init = np.array([[0.0, -5.0]]).T
    fig, ax = plot_iters_sbm(data, x_init, fig, ax)
    ax.set_title("Spectral Bundle Method")

    # CUT PLANE
    # plot objective
    fig, ax = plot_obj(data=data, rng=r)
    # plot iterations
    fig, ax = plot_iters(data, x_init, fig, ax, method="cuts")
    ax.set_title("Cut Plane")

    # SUBGRAD
    # plot objective
    fig, ax = plot_obj(data=data, rng=r)
    # plot iterations
    fig, ax = plot_iters(data, x_init, fig, ax, method="sub")
    ax.set_title("Subgrad")

    plt.show()


if __name__ == "__main__":
    # run_test_prob()
    # run_tent_prob()
    run_sphere_prob()

    print("done")
