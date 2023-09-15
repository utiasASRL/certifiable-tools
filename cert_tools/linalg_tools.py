import numpy as np
import scipy.linalg as la

import sparseqr as sqr

METHOD = "qrp"
NULL_THRESH = 1e-5


def rank_project(X, p=1, tolerance=1e-10):
    """Project symmetric matrix X to matrix of rank p."""
    assert la.issymmetric(X)
    E, V = np.linalg.eigh(X)
    if p is None:
        p = np.sum(np.abs(E) > tolerance)
    x = V[:, -p:] * np.sqrt(E[-p:])

    X_hat = np.outer(x, x)
    info = {"error X": np.linalg.norm(X_hat - X), "error eigs": np.sum(np.abs(E[:p]))}
    return x, info


def find_dependent_columns(A_sparse, tolerance=1e-10):
    """
    Returns a list of indices corresponding to the columns of A_sparse that are linearly dependent.
    """
    # Use sparse rank revealing QR
    # We "solve" a least squares problem to get the rank and permutations
    # This is the cheapest way to use sparse QR, since it does not require
    # explicit construction of the Q matrix. We can't do this with qr function
    # because the "just return R" option is not exposed.
    Z, R, E, rank = sqr.rz(
        A_sparse, np.zeros((A_sparse.shape[0], 1)), tolerance=tolerance
    )

    # Sort the diagonal values. Note that SuiteSparse uses A_sparseMD/METIS ordering
    # to acheive sparsity.
    r_vals = np.abs(R.diagonal())
    sort_inds = np.argsort(r_vals)[::-1]
    if rank < A_sparse.shape[1]:
        print(f"clean_constraints: keeping {rank}/{A_sparse.shape[1]} independent")

    bad_idx = list(range(A_sparse.shape[1]))
    keep_idx = sorted(E[sort_inds[:rank]])[::-1]
    for good_idx in keep_idx:
        del bad_idx[good_idx]

    # Sanity check
    Z, R, E, rank_full = sqr.rz(
        A_sparse.tocsc()[:, keep_idx],
        np.zeros((A_sparse.shape[0], 1)),
        tolerance=tolerance,
    )
    if rank_full != rank:
        print(
            f"Warning: selected constraints did not pass lin. independence check. Rank is {rank_full}, should be {rank}."
        )
    return bad_idx


def get_nullspace(A_dense, method=METHOD, tolerance=NULL_THRESH):
    if method == "svd":
        U, S, Vh = np.linalg.svd(
            A_dense
        )  # nullspace of A_dense is in last columns of V / last rows of Vh
        rank = np.sum(np.abs(S) > tolerance)
        basis = Vh[rank:, :]

    elif method == "qr":
        # if A_dense.T = QR, the last n-r columns
        # of R make up the nullspace of A_dense.
        Q, R = np.linalg.qr(A_dense.T)
        S = np.abs(np.diag(R))
        sorted_idx = np.argsort(S)[::-1]
        S = S[sorted_idx]
        rank = np.where(S < tolerance)[0][0]
        # decreasing order
        basis = Q[:, sorted_idx[rank:]].T
    elif method == "qrp":
        # Based on Section 5.5.5 "Basic Solutions via QR with Column Pivoting" from Golub and Van Loan.

        assert A_dense.shape[0] >= A_dense.shape[1], "only tall matrices supported"

        Q, R, p = la.qr(A_dense, pivoting=True, mode="economic")
        S = np.abs(np.diag(R))
        rank = np.sum(S > tolerance)
        R1, R2 = R[:rank, :rank], R[:rank, rank:]
        # [R1  R2]  @  [R1^-1 @ R2] = [R2 - R2]
        # [0   0 ]     [    -I    ]   [0]
        N = np.vstack([la.solve_triangular(R1, R2), -np.eye(R2.shape[1])])

        basis = np.zeros(N.T.shape)
        basis[:, p] = N.T
    else:
        raise ValueError(method)

    # test that it is indeed a null space
    error = A_dense @ basis.T
    info = {"values": S, "error": error}
    return basis, info
