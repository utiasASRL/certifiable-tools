from copy import deepcopy

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

METHOD = "qrp"
NULL_THRESH = 1e-5


def svec(S, order="C"):
    """Convert symmetric matrix to vectorized form

    Args:
        mat (_type_): matrix to vectorize
        order: 'C' or 'R' depending on whether the vectorization uses row-major or column-major ordering after isolating the upper triagle
    """
    if sp.issparse(S):
        L = sp.triu(S)
        rows, cols = L.nonzero()
        vals = L.data
        vals[rows < cols] *= np.sqrt(2)
        N = S.shape[0]
        vec_size = np.int32(N * (N + 1) / 2)
        if order == "R":
            row_offs = rows * N - rows * (rows - 1) / 2
            inds = (cols - rows + row_offs).astype(np.int32)
        elif order == "C":
            inds = (cols * (cols + 1) / 2 + rows).astype(np.int32)
        zero = np.zeros(len(inds)).astype(np.int32)
        return sp.csc_array((vals, (zero, inds)), shape=(1, vec_size))
    else:
        n = S.shape[0]
        S = np.copy(S)
        S *= np.sqrt(2)
        S[range(n), range(n)] /= np.sqrt(2)
        if order == "R":
            vec = S[np.triu_indices(n)]
        elif order == "C":
            triu_indices = np.triu_indices(n)
            sorted_indices = np.lexsort(triu_indices)
            vec = S[triu_indices][sorted_indices]
        else:
            raise ValueError("Invalid order")
        return vec


def smat(s, order="C"):
    """get symmetric matrix from vectorized form

    Args:
        s (_type_): vector to matricize
        order: 'C' or 'R' depending on whether the vectorization uses row-major or column-major ordering after isolating the upper triagle
    """
    if sp.issparse(s):
        s = s.toarray().squeeze(0)
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    if order == "R":
        S[np.triu_indices(n)] = s
    elif order == "C":
        triu_indices = np.triu_indices(n)
        sorted_indices = np.lexsort(triu_indices)
        indices = (triu_indices[0][sorted_indices], triu_indices[1][sorted_indices])
        S[indices] = s
    else:
        raise ValueError("Invalid order")
    S /= np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S


def project_so3(X):
    if X.shape[0] == 4:
        X = deepcopy(X)
        rot = X[:3, :3]
        U, S, Vh = np.linalg.svd(rot)
        rot = U @ Vh
        X[:3, :3] = rot
        return X
    else:
        U, S, Vh = np.linalg.svd(X)
        return U @ Vh


def rank_project(X, p=1, tolerance=1e-10):
    """Project matrix X to matrix of rank p."""
    try:
        assert la.issymmetric(X, atol=tolerance)
        E, V = np.linalg.eigh(X)
        if p is None:
            p = np.sum(np.abs(E) > tolerance)
        x = V[:, -p:] * np.sqrt(E[-p:])

        if p == 1:
            X_hat = np.outer(x, x)
        else:
            X_hat = x @ x.T
        info = {
            "error X": np.linalg.norm(X_hat - X),
            "error eigs": np.sum(np.abs(E[:-p])),
            "EVR": abs(E[-p] / E[-p - 1]),  # largest over second-largest
        }
    except (ValueError, AssertionError):
        U, E, Vh = np.linalg.svd(X)
        if p is None:
            p = np.sum(np.abs(E) > tolerance)
        X_hat = U[:, :p] @ np.diag(E[:p]) @ Vh[:p, :]
        x = U[:, :p] @ np.diag(E[:p])
        info = {
            "error X": np.linalg.norm(X_hat - X),
            "error eigs": np.sum(np.abs(E[p:])),
            "EVR": abs(E[p - 1] / E[p]),  # largest over second-largest
        }
    return x, info


def find_dependent_columns(A_sparse, tolerance=1e-10, verbose=False, debug=False):
    """
    Returns a list of indices corresponding to the columns of A_sparse that are linearly dependent.
    """
    import sparseqr as sqr

    # Use sparse rank revealing QR
    # We "solve" a least squares problem to get the rank and permutations. This is the cheapest way to use sparse QR, since it does not require explicit construction of the Q matrix. We can't do this with qr function because the "just return R" option is not exposed.
    Z, R, E, rank = sqr.rz(
        A_sparse, np.zeros((A_sparse.shape[0], 1)), tolerance=tolerance
    )
    if rank == A_sparse.shape[1]:
        return []

    # Sort the diagonal values. Note that SuiteSparse uses AMD/METIS ordering to acheive sparsity.
    r_vals = np.abs(R.diagonal())
    sort_inds = np.argsort(r_vals)[::-1]
    if (rank < A_sparse.shape[1]) and verbose:
        print(f"clean_constraints: keeping {rank}/{A_sparse.shape[1]} independent")

    bad_idx = list(range(A_sparse.shape[1]))
    good_idx_list = sorted(E[sort_inds[:rank]])[::-1]
    for good_idx in good_idx_list:
        del bad_idx[good_idx]

    # Sanity check
    if debug:
        Z, R, E, rank_full = sqr.rz(
            A_sparse.tocsc()[:, good_idx_list],
            np.zeros((A_sparse.shape[0], 1)),
            tolerance=tolerance,
        )
        if rank_full != rank:
            print(
                f"Warning: selected constraints did not pass lin. independence check. Rank is {rank_full}, should be {rank}."
            )
    return bad_idx


def get_nullspace(A_dense, method=METHOD, tolerance=NULL_THRESH):
    info = {}

    if method != "qrp":
        print("Warning: method other than qrp is not recommended.")

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
        # assert A_dense.shape[0] >= A_dense.shape[1], "only tall matrices supported"

        Q, R, P = la.qr(A_dense, pivoting=True, mode="economic")
        if Q.shape[0] < 1e4:
            np.testing.assert_allclose(Q @ R, A_dense[:, P], atol=1e-5)

        S = np.abs(np.diag(R))
        rank = np.sum(S > tolerance)
        R1 = R[:rank, :]
        R11, R12 = R1[:, :rank], R1[:, rank:]
        # [R11  R12]  @  [R11^-1 @ R12] = [R12 - R12]
        # [0    0 ]       [    -I    ]    [0]
        N = np.vstack([la.solve_triangular(R11, R12), -np.eye(R12.shape[1])])

        # Inverse permutation
        Pinv = np.zeros(len(P), int)
        for k, p in enumerate(P):
            Pinv[p] = k
        LHS = R1[:, Pinv]

        info["Q1"] = Q[:, :rank]
        info["LHS"] = LHS

        basis = np.zeros(N.T.shape)
        basis[:, P] = N.T
    else:
        raise ValueError(method)

    # test that it is indeed a null space
    error = A_dense @ basis.T
    info["values"] = S
    info["error"] = error
    return basis, info
