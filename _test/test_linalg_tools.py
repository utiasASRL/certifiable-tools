import unittest

import numpy as np
import scipy.sparse as sp

from cert_tools.linalg_tools import rank_project, smat, svec


def best_rotation_nd(X, Y):
    """
    X, Y: (N, d) arrays with corresponding row points.
    Returns: R (dxd orthogonal matrix),
    such that Y ≈ (X @ R.T)
    """
    cx = X.mean(axis=0)
    cy = Y.mean(axis=0)
    Xc = X - cx
    Yc = Y - cy

    H = Xc.T @ Yc  # d x d
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    return R


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
        prod_vec = np.dot(s1, s2)  # type: ignore
        assert abs(prod_mat - prod_vec) < 1e-10, "PSD Inner product not equal"

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
        np.testing.assert_almost_equal(s1_dense, s1.toarray().squeeze(0))  # type: ignore
        np.testing.assert_almost_equal(s2_dense, s2.toarray().squeeze(0))  # type: ignore
        # test mapping
        np.testing.assert_almost_equal(smat(s1), S1.toarray())
        np.testing.assert_almost_equal(smat(s2), S2.toarray())
        # products should be equal
        prod_mat = np.trace(S1.toarray() @ S2.toarray())
        prod_vec = (s1 @ s2.T).toarray()  # type: ignore
        assert abs(prod_mat - prod_vec) < 1e-10, "PSD Inner product not equal"

    def test_rank_project(self):
        # test symmetric rank-one matrix
        x_gt = np.random.randn(5, 1)
        X = x_gt @ x_gt.T
        x_test, info_rank = rank_project(X, p=1)
        if x_gt[0] * x_test[0] < 0:
            x_test = -x_test
        np.testing.assert_allclose(x_test, x_gt)

        # test symmatric rank-two matrix
        # below we need to compensate for a rotation because
        # X = U E U' = U sqrt(E) sqrt(E) U' = U sqrt(E) R' R sqrt(E) U'
        # for any orthogonal R.
        x_gt = np.random.randn(5, 2)
        X = x_gt @ x_gt.T
        x_test, info_rank = rank_project(X, p=2)
        R = best_rotation_nd(x_test, x_gt)
        x_test = x_test @ R.T  # align for comparison
        np.testing.assert_allclose(x_test, x_gt)

        # test non-symmetric rank-one matrix
        x_gt = np.random.randn(5, 1)
        X = x_gt @ x_gt.T
        X[0, 2] += 1e-8
        x_test, info_rank = rank_project(X, p=1)
        if x_gt[0] * x_test[0] < 0:
            x_test = -x_test
        np.testing.assert_allclose(x_test, x_gt, atol=1e-5)

        # test non-symmetric rank-two matrix
        x_gt = np.random.randn(5, 2)
        X = x_gt @ x_gt.T
        X[0, 2] += 1e-8
        x_test, info_rank = rank_project(X, p=2)
        R = best_rotation_nd(x_test, x_gt)
        x_test = x_test @ R.T  # align for comparison
        np.testing.assert_allclose(x_test, x_gt, atol=1e-5)


if __name__ == "__main__":
    test = TestLinAlg()
    test.test_svec()
    test.test_svec_sparse()
    test.test_rank_project()
