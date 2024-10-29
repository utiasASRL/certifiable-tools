import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from cert_tools import HomQCQP
from cert_tools.linalg_tools import smat, svec
from cert_tools.problems.rot_synch import RotSynchLoopProblem
from cert_tools.sdp_solvers import solve_sdp_homqcqp
from cert_tools.sparse_solvers import solve_clarabel, solve_dsdp


def get_chain_rot_prob(N=10):
    return RotSynchLoopProblem(N=N, loop_pose=-1, locked_pose=0)


def get_loop_rot_prob():
    return RotSynchLoopProblem()


class TestHomQCQP(unittest.TestCase):

    def test_solve(self):
        # Create chain of rotations problem:
        problem = get_chain_rot_prob()
        assert isinstance(problem, HomQCQP), TypeError(
            "Problem should be homogenized qcqp object"
        )
        # Solve SDP via standard method
        X, info, time = solve_sdp_homqcqp(problem, verbose=True)
        # Convert solution
        R = problem.convert_sdp_to_rot(X)

    def test_get_asg(self, plot=False):
        """Test retrieval of aggregate sparsity graph"""
        # Test on chain graph
        problem = get_chain_rot_prob()
        problem.clique_decomposition()
        # No fill in expected
        assert problem.symb.fill[0] == 0, ValueError("Expected no fill in")
        if plot:
            problem.plot_asg()

        # Test on loop graph
        problem = get_loop_rot_prob()
        problem.get_asg(rm_homog=False)
        problem.clique_decomposition()
        assert problem.symb.fill[0] > 0, ValueError("Expected fill in")
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
            assert set(clique.separator).issubset(vertices), ValueError(
                "separator set should be in set of involved clique vertices"
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
            # separator
            sepset = clique_l.separator
            if len(sepset) == 0:  # skip the root clique
                continue
            # fet parent clique and separator set
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
        """Test solve of Decomposed SDP using interior point solver.
        Test minimum rank SDP completion."""
        # Test chain topology
        nvars = 5
        problem = get_chain_rot_prob(N=nvars)
        problem.clique_decomposition()  # get cliques
        # Solve non-decomposed problem
        X, info, time = solve_sdp_homqcqp(problem, verbose=True)
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

        # Test PSD completion
        # Perform completion
        Y, factor_dict = problem.get_mr_completion(c_list)
        # Check locked pose
        R_0 = factor_dict["0"].reshape((3, 3)).T
        np.testing.assert_allclose(
            problem.R_gt[0],
            R_0,
            atol=1e-7,
            err_msg="Locked pose incorrect after PSD Completion",
        )
        X_complete = Y @ Y.T
        np.testing.assert_allclose(
            X,
            X_complete,
            atol=1e-7,
            err_msg="Completed and non-decomposed solutions differ",
        )

    def test_standard_form(self):
        """Test that the standard form problem definition is correct"""
        nvars = 2
        problem = get_chain_rot_prob(N=nvars)
        problem.get_asg()
        P, q, A, b = problem.get_standard_form()

        # get solution from MOSEK
        X, info, time = solve_sdp_homqcqp(problem, verbose=True)
        x = svec(X)

        # Check cost matrix
        cost = np.dot(b, x)
        np.testing.assert_allclose(
            cost, info["cost"], atol=1e-12, err_msg="Cost incorrect"
        )
        # Check constraints
        for i, vec in enumerate(A.T):
            a = vec.toarray().squeeze(0)
            value = np.dot(a, x)
            np.testing.assert_allclose(
                value, -q[i], atol=1e-10, err_msg=f"Constraint {i} has violation"
            )

    def test_clarabel(self):
        nvars = 2
        problem = get_chain_rot_prob(N=nvars)
        problem.get_asg()
        X_clarabel = solve_clarabel(problem)
        X, info, time = solve_sdp_homqcqp(problem, verbose=True)

        np.testing.assert_allclose(
            X_clarabel,
            X,
            atol=1e-9,
            err_msg="Clarabel and MOSEK solutions differ",
        )


if __name__ == "__main__":
    test = TestHomQCQP()
    # test.test_solve()
    # test.test_get_asg(plot=True)
    # test.test_clique_decomp(plot=False)
    test.test_consistency_constraints()
    # test.test_decompose_matrix()
    # test.test_solve_dsdp()
    # test.test_standard_form()
    # test.test_clarabel()
