import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.problems.rot_synch import RotSynchLoopProblem
from cert_tools.sparse_solvers import solve_dsdp, solve_oneshot


def get_chain_rot_prob(N=10):
    return RotSynchLoopProblem(N=N, loop_pose=-1, locked_pose=0)


def get_loop_rot_prob():
    return RotSynchLoopProblem()


class TestHomQCQP(unittest.TestCase):

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

    def test_build_jtree(rm_homog=False, plot=False):
        """Test function that builds the junction tree associated
        with the problem"""

        # Test chain topology
        problem = get_chain_rot_prob()
        problem.get_asg(rm_homog=rm_homog)  # Get Graph
        problem.triangulate_graph()  # Triangulate graph
        problem.build_jtree()  # Build Junction tree
        if plot:
            # problem.plot_asg()
            problem.plot_jtree()
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

        # Check that mapping from variables to cliques is correct
        cliques = problem.jtree.vs
        # loop through vars
        for varname in problem.var_sizes.keys():
            # Skip homogenizing var
            if rm_homog and varname == "h":
                continue
            # loop through map values
            for i in problem.var_clique_map[varname]:
                # check that var is actually in clique
                assert varname in cliques[i]["vlist"]

    def test_interclique_constraints():
        # Test chain topology
        nvars = 5
        problem = get_chain_rot_prob(N=nvars)
        problem.get_asg(rm_homog=False)  # Get Graph
        problem.triangulate_graph()  # Triangulate graph
        problem.build_jtree()  # Build Junction tree
        eq_list = problem.build_interclique_constraints()

        # check the number of constraints generated
        clq_dim = 10  # homogenizing var plus rotation
        n_cons_per_sep = round(clq_dim * (clq_dim + 1) / 2)
        assert len(eq_list) == (nvars - 2) * n_cons_per_sep, ValueError(
            "Wrong number of equality constraints"
        )

        # Run problem with no other costs and constraints
        problem.C *= 0.0
        problem.As = []
        x_list, info = solve_dsdp(problem, verbose=True, tol=1e-8)
        # Verify that the clique variables are equal on overlaps
        for edge in problem.jtree.es:
            # Get clique objects and seperator set
            k = edge.tuple[0]
            l = edge.tuple[1]
            clique_k = problem.jtree.vs[k]["clique"]
            clique_l = problem.jtree.vs[l]["clique"]
            sepset = edge["sepset"]
            X_k = clique_k.get_slices(x_list[k], sepset)
            X_l = clique_k.get_slices(x_list[l], sepset)
            np.testing.assert_allclose(
                X_k, X_l, atol=1e-9, err_msg=f"Clique overlaps not equal for ({k},{l})"
            )

    def test_decompose_matrix(self):
        # setup
        nvars = 5
        problem = get_chain_rot_prob(N=nvars)
        problem.get_asg(rm_homog=False)  # Get Graph
        problem.triangulate_graph()  # Triangulate graph
        problem.build_jtree()  # Build Junction tree
        C = problem.C

        # args
        self.assertRaises(ValueError, problem.decompose_matrix, C, method="banana")
        A = np.zeros((4, 4))
        self.assertRaises(AssertionError, problem.decompose_matrix, A)

        # functionality
        for method in ["split", "first"]:
            C_d = problem.decompose_matrix(C, method=method)
            assert len(C_d.keys()) == nvars - 1, ValueError(
                f"{method} Method: Wrong number of cliques in decomposed matrix"
            )
            assert len(set(C_d.keys())) == len(C_d.keys()), ValueError(
                f"{method} Method: Keys of decomposed matrix are not unique"
            )
            # reconstruct matrix
            mat = PolyMatrix()
            for pmat in C_d.values():
                mat += pmat
            # Get numerical matrices
            C_mat = C.get_matrix(problem.var_sizes).todense()
            C_d_mat = mat.get_matrix(problem.var_sizes).todense()
            np.testing.assert_allclose(
                C_mat,
                C_d_mat,
                atol=1e-12,
                err_msg="Clique decomposition then reassembly failed for objective",
            )


if __name__ == "__main__":
    test = TestHomQCQP()
    # test.test_solve()
    # test.test_get_asg(plot=True)
    # test.test_build_jtree(plot=True)
    # test.test_interclique_constraints()
    test.test_decompose_matrix()
