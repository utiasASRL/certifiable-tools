from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.problems.rot_synch import RotSynchLoopProblem
from cert_tools.sparse_solvers import solve_oneshot


def get_chain_rot_prob():
    return RotSynchLoopProblem(loop_pose=-1, locked_pose=0)


def test_solve():
    # Create chain of rotations problem:
    problem = get_chain_rot_prob()
    assert isinstance(problem, HomQCQP), TypeError(
        "Problem should be homogenized qcqp object"
    )
    # Solve SDP via standard method
    X, info, time = problem.solve_sdp(verbose=True)
    # Convert solution
    R = problem.convert_sdp_to_rot(X)


def test_get_aggspar_graph():
    problem = get_chain_rot_prob()
    problem.get_aggspar_graph()


if __name__ == "__main__":
    test_solve()
