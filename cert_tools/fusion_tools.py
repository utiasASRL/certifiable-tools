import numpy as np
import scipy.sparse as sp
from mosek.fusion import Matrix


def mat_fusion(X):
    """Convert sparse matrix X to fusion format"""
    try:
        X.eliminate_zeros()
    except AttributeError:
        X = sp.csr_array(X)
    I, J = X.nonzero()
    I = I.astype(np.int32)
    J = J.astype(np.int32)
    V = X.data.astype(np.double)
    return Matrix.sparse(*X.shape, I, J, V)


def get_slice(X: Matrix, i: int):
    (N, X_dim, X_dim) = X.getShape()
    return X.slice([i, 0, 0], [i + 1, X_dim, X_dim]).reshape([X_dim, X_dim])


def svec_fusion(X):
    """Not working"""
    N = X.numRows()
    assert isinstance(X, Matrix)
    return [X.index(i, j) for (i, j) in zip(*np.triu_indices(N))]


# TODO(FD) not used anymore as now we are setting accSolutionStatus to Anything.
# Before, this was used to read from UNKNOWN problem status.
def read_costs_from_mosek(fname):
    f = open(fname, "r")
    ls = f.readlines()
    primal_line = ls[-2].split(" ")
    assert "Primal." in primal_line
    primal_value = float(primal_line[primal_line.index("obj:") + 1])

    dual_line = ls[-1].split(" ")
    assert "Dual." in dual_line
    dual_value = float(dual_line[dual_line.index("obj:") + 1])
    return primal_value, dual_value
