import os

import numpy as np

from cert_tools.sdp_solvers import (
    solve_low_rank_sdp,
    solve_sdp_mosek,
    solve_feasibility_sdp,
)
from cert_tools.eopt_solvers import solve_eopt_cuts
from cert_tools.eopt_solvers_qp import solve_eopt_qp

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")

if __name__ == "__main__":
    import pickle

    # no redundant constraints
    # fname = os.path.join(root_dir, "_test", "test_prob_10Gc.pkl")

    # with redundant constraints
    fname = os.path.join(root_dir, "_test", "test_prob_11Gc.pkl")

    global_min = True

    with open(fname, "rb") as f:
        data = pickle.load(f)

    # solve_low_rank_sdp(**data)
    # solve_sdp_mosek(**data)

    H, info_feas = solve_feasibility_sdp(**data, adjust=False)
    info_cuts = solve_eopt_cuts(**data)  # , x_init=np.array(info_feas["yvals"]))
    info_qp = solve_eopt_qp(**data, verbose=2)
    print("done")
