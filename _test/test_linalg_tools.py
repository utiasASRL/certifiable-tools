import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from cert_tools.linalg_tools import smat, svec


class TestLinAlg(unittest.TestCase):

    def get_psd_mat(self, n=3):
        # get random symmetric matrix
        A = np.random.random((n, n))
        U, S, V = np.linalg.svd(A)
        S = np.abs(S)
        return (U * S) @ U.T

    def test_svec(self):
        # fix seed
        np.random.seed(0)
        # generate matrices
        S1 = self.get_psd_mat()
        S2 = self.get_psd_mat()
        # Vectorize
        s1 = svec(S1)
        s2 = svec(S2)
        # test mapping
        np.testing.assert_almost_equal(smat(s1), S1)
        np.testing.assert_almost_equal(smat(s2), S2)
        # products should be equal
        prod_mat = np.trace(S1 @ S2)
        prod_vec = np.dot(s1, s2)
        assert prod_mat == prod_vec, "PSD Inner product not equal"

    def test_svec_sparse(self):
        # fix seed
        np.random.seed(0)
        # generate matrices
        S1_dense = self.get_psd_mat()
        S2_dense = self.get_psd_mat()
        # Remove element to make sure still works when not dense
        # S1_dense[4, 5] = 0.0
        # S1_dense[5, 4] = 0.0
        S1 = sp.csc_matrix(S1_dense)
        S2 = sp.csc_matrix(S2_dense)
        S1.eliminate_zeros()
        S2.eliminate_zeros()
        # Vectorize
        s1 = svec(S1)
        s2 = svec(S2)
        s1_dense = svec(S1_dense)
        s2_dense = svec(S2_dense)
        np.testing.assert_almost_equal(s1_dense, s1.toarray().squeeze(0))
        np.testing.assert_almost_equal(s2_dense, s2.toarray().squeeze(0))
        # test mapping
        np.testing.assert_almost_equal(smat(s1), S1.toarray())
        np.testing.assert_almost_equal(smat(s2), S2.toarray())
        # products should be equal
        prod_mat = np.trace(S1.toarray() @ S2.toarray())
        prod_vec = (s1 @ s2.T).toarray()
        assert prod_mat == prod_vec, "PSD Inner product not equal"


if __name__ == "__main__":
    test = TestLinAlg()
    test.test_svec()
    test.test_svec_sparse()
