import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.problems.rot_synch import RotSynchLoopProblem
from cert_tools.sparse_solvers import solve_dsdp


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

    def test_clique_decomp(self, rm_homog=False, plot=False):
        """Test solve of Decomposed SDP using interior point solver"""
        # Test chain topology
        nvars = 5
        problem = get_chain_rot_prob(N=nvars)
        problem.clique_decomposition()
        if plot:
            problem.plot_asg()
            problem.plot_ctree()
        # Check number of cliques
        assert len(problem.cliques) == nvars - 1, ValueError(
            "Junction tree has wrong number of cliques"
        )
        # Test clique sizes
        if rm_homog:
            cliquesize = 2
        else:
            cliquesize = 3
        for clique in problem.cliques:
            assert len(clique.var_sizes.keys()) == cliquesize, ValueError(
                "Cliques should have 3 vertices each"
            )

        for clique in problem.cliques:
            parent = problem.cliques[clique.parent]

            vertices = list(parent.var_sizes.keys()) + list(clique.var_sizes.keys())
            assert set(clique.seperator).issubset(vertices), ValueError(
                "seperator set should be in set of involved clique vertices"
            )

        # Check that mapping from variables to cliques is correct
        cliques = problem.cliques
        # loop through vars
        for varname in problem.var_sizes.keys():
            # Skip homogenizing var
            if rm_homog and varname == "h":
                continue
            # loop through map values
            for i in problem.var_clique_map[varname]:
                # check that var is actually in clique
                assert varname in cliques[i].var_sizes.keys()

    def test_consistency_constraints(self):
        """Test clique overlap consistency constraints"""
        # Test chain topology
        nvars = 5
        problem = get_chain_rot_prob(N=nvars)
        problem.get_asg(rm_homog=False)  # Get Graph
        problem.clique_decomposition()  # Run clique decomposition
        eq_list = problem.get_consistency_constraints()

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
        for l, clique_l in enumerate(problem.cliques):
            # seperator
            sepset = clique_l.seperator
            if len(sepset) == 0:  # skip the root clique
                continue
            # fet parent clique and seperator set
            k = clique_l.parent
            clique_k = problem.cliques[k]

            # get variables
            X_k = clique_k.get_slices(x_list[k], sepset)
            X_l = clique_k.get_slices(x_list[l], sepset)
            np.testing.assert_allclose(
                X_k, X_l, atol=1e-9, err_msg=f"Clique overlaps not equal for ({k},{l})"
            )

    def test_decompose_matrix(self):
        """Test matrix decomposition into cliques"""
        # setup
        nvars = 5
        problem = get_chain_rot_prob(N=nvars)
        problem.get_asg(rm_homog=False)  # Get Graph
        problem.clique_decomposition()  # get clique decomposition
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

    def test_solve_dsdp(self):
        """Test solve of Decomposed SDP using interior point solver"""
        # Test chain topology
        nvars = 5
        problem = get_chain_rot_prob(N=nvars)
        problem.get_asg(rm_homog=False)  # Get agg sparse graph
        problem.clique_decomposition()  # get cliques
        # Solve non-decomposed problem
        X, info, time = problem.solve_sdp(verbose=True)
        # get cliques from non-decomposed solution
        c_list_nd = problem.get_cliques_from_psd_mat(X)
        # Solve decomposed problem (Interior Point Version)
        c_list, info = solve_dsdp(problem, verbose=True, tol=1e-8)  # check solutions
        for c, c_nd in zip(c_list, c_list_nd):
            np.testing.assert_allclose(
                c,
                c_nd,
                atol=1e-7,
                err_msg="Decomposed and non-decomposed solutions differ",
            )

        # Test solution recovery
        # X_complete = problem.get_psd_completion(c_list)


if __name__ == "__main__":
    test = TestHomQCQP()
    # test.test_solve()
    # test.test_get_asg(plot=True)
    # test.test_clique_decomp(plot=False)
    # test.test_consistency_constraints()
    # test.test_decompose_matrix()
    test.test_solve_dsdp()
