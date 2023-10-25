import os

import numpy as np
import matplotlib.pylab as plt

from cert_tools.sdp_solvers import solve_feasibility_sdp
from cert_tools.eopt_solvers import solve_eopt
from cert_tools.eopt_solvers_qp import solve_eopt_qp

np.set_printoptions(precision=2)

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def save_all(figname):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    def multipage(filename, figs=None, dpi=200):
        pp = PdfPages(filename)
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format="pdf")
        pp.close()

    multipage(figname)


if __name__ == "__main__":
    import pickle

    exploit_centered = False

    problem_name = f"test_prob_11G"  # with redundant constraints
    # problem_name = "test_prob_10Gc" # no redundant constraints

    fname = os.path.join(root_dir, "_test", f"{problem_name}.pkl")

    with open(fname, "rb") as f:
        data = pickle.load(f)

    # solve_low_rank_sdp(**data)
    # solve_sdp_mosek(**data)

    H, info_feas = solve_feasibility_sdp(**data, adjust=False)
    eigs = np.linalg.eigvalsh(H.toarray())[:3]

    fig, ax = plt.subplots()
    ax.matshow(H.toarray())
    ax.set_title(f"H \n{eigs}")
    print("minimum eigenvalues:", eigs)

    info_cuts = solve_eopt(
        **data, exploit_centered=exploit_centered, plot=True
    )  # , x_init=np.array(info_feas["yvals"]))
    # info_qp = solve_eopt_qp(**data, verbose=2)
    plt.show()

    figname = os.path.join(root_dir, "_plots", f"{problem_name}.pdf")
    save_all(figname)
    print(f"saved all as {figname}")
