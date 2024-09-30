from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.problems.rot_synch import RotSynchLoopProblem
from cert_tools.sparse_solvers import solve_oneshot


def get_chain_rot_prob():
    return RotSynchLoopProblem(N=10, loop_pose=-1, locked_pose=0)


def get_loop_rot_prob():
    return RotSynchLoopProblem()


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


def test_get_asg(plot=False):
    """Test retreival of aggregate sparsity graph"""
    # Test on chain graph
    problem = get_chain_rot_prob()
    problem.get_asg(rm_homog=True)  # Get Graph
    problem.triangulate_graph()  # Triangulate graph
    # No fill in expected
    assert any(problem.asg.es["fill_edge"]) is False
    if plot:
        problem.plot_asg()

    # Test on loop graph
    problem = get_loop_rot_prob()
    problem.get_asg(rm_homog=True)
    problem.triangulate_graph()
    assert any(problem.asg.es["fill_edge"]) is True
    if plot:
        problem.plot_asg()


def test_build_jtree(rm_homog=True, plot=False):
    """Test function that builds the junction tree associated
    with the problem"""

    # Test chain topology
    problem = get_chain_rot_prob()
    problem.get_asg(rm_homog=rm_homog)  # Get Graph
    problem.triangulate_graph()  # Triangulate graph
    problem.build_jtree()  # Build Junction tree

    # Check number of cliques
    assert len(problem.jtree.vs) == 9, ValueError(
        "Junction tree has wrong number of cliques"
    )
    # Test clique sizes
    if rm_homog:
        cliquesize = 2
    else:
        cliquesize = 3
    for clique in problem.jtree.vs:
        assert len(clique["vlist"]) == cliquesize, ValueError(
            "Cliques should have 3 vertices each"
        )

    for jedge in problem.jtree.es:
        cliques = problem.jtree.vs.select(jedge.tuple)
        vertices = set([v for v_list in cliques["vlist"] for v in v_list])
        assert set(jedge["sepset"]).issubset(vertices), ValueError(
            "seperator set should be in set of involved clique vertices"
        )

    if plot:
        problem.plot_jtree()


if __name__ == "__main__":
    # test_solve()
    # test_get_asg(plot=True)
    test_build_jtree(plot=True)
