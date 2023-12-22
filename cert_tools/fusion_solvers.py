import sys
import numpy as np

from mosek.fusion import Domain, Expr, ObjectiveSense, Model
from mosek.fusion import Matrix

from cert_tools.sdp_solvers import adjust_Q


def mat_fusion(X):
    """Convert sparse matrix X to fusion format"""
    X.eliminate_zeros()
    I, J = X.nonzero()
    V = X.data
    return Matrix.sparse(*X.shape, I, J, V)


def get_slice(X, i):
    (N, X_dim, X_dim) = X.getShape()
    return X.slice([i, 0, 0], [i + 1, X_dim, X_dim]).reshape([X_dim, X_dim])


def solve_sdp_fusion(Q, Constraints, adjust=False, verbose=False, use_primal=False):
    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    if use_primal:
        with Model("primal") as M:
            # creates (N x X_dim x X_dim) variable
            X = M.variable("X", Domain.inPSDCone(Q.shape[0]))

            # standard equality constraints
            for A, b in Constraints:
                M.constraint(Expr.dot(mat_fusion(A), X), Domain.equalsTo(b))

            M.objective(ObjectiveSense.Minimize, Expr.dot(mat_fusion(Q_here), X))

            # M.setSolverParam("intpntCoTolRelGap", 1.0e-7)
            if verbose:
                M.setLogHandler(sys.stdout)
            M.solve()

            X = np.reshape(X.level(), Q.shape)
            cost = M.primalObjValue() * scale + offset
            info = {"success": True, "cost": cost}
    else:
        # TODO(FD) below is extremely slow and runs out of memory for 200 x 200 matrices.
        with Model("dual") as M:
            # creates (N x X_dim x X_dim) variable
            m = len(Constraints)
            b = np.array([-b for A, b in Constraints])[None, :]
            y = M.variable("y", [m, 1])

            # standard equality constraints
            con = M.constraint(
                Expr.add(
                    mat_fusion(Q_here),
                    Expr.add(
                        [
                            Expr.mul(mat_fusion(Constraints[i][0]), y.index([i, 0]))
                            for i in range(m)
                        ]
                    ),
                ),
                Domain.inPSDCone(Q.shape[0]),
            )
            M.objective(ObjectiveSense.Maximize, Expr.sum(Expr.mul(Matrix.dense(b), y)))

            # M.setSolverParam("intpntCoTolRelGap", 1.0e-7)
            if verbose:
                M.setLogHandler(sys.stdout)
            M.solve()

            X = np.reshape(con.dual(), Q.shape)
            cost = M.primalObjValue() * scale + offset
            info = {"success": True, "cost": cost}
    return X, info
