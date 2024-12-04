import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from cert_tools.base_clique import BaseClique
from cert_tools.hom_qcqp import HomQCQP
from cert_tools.sdp_solvers import adjust_Q

CONSTRAIN_ALL_OVERLAP = False


def update_rho(rho, dual_res, primal_res, mu, tau):
    """Update rho as suggested by [Boyd 2010]."""
    assert tau >= 1.0
    assert mu >= 0
    if np.ndim(rho) > 0:
        rho[np.where(primal_res >= mu * dual_res)[0]] *= tau
        rho[np.where(dual_res >= mu * primal_res)[0]] /= tau
        return rho
    else:
        if primal_res >= mu * dual_res:
            return rho * tau
        elif dual_res >= mu * primal_res:
            return rho / tau
        else:
            return rho


def initialize_overlap(clique_list):
    for k, clique in enumerate(clique_list):
        assert isinstance(clique, ADMMClique)

        # below are for solving with fusion API
        clique.generate_overlap_slices(N=len(clique_list))
        clique.generate_F(N=len(clique_list))

        # below are for solving with cvxpy
        left = clique_list[k - 1].X_var if k > 0 else None
        right = clique_list[k + 1].X_var if k < len(clique_list) - 1 else None
        clique.g = clique.generate_g(left=left, right=right)
        assert clique.F.shape[0] == clique.g.shape[0]


class ADMMClique(BaseClique):
    def __init__(
        self,
        Q,
        Constraints: list,
        var_dict: dict,
        index,
        X: np.ndarray = None,
        hom="h",
    ):
        self.Q = Q
        self.Constraints = Constraints

        self.X_dim = Q.get_shape()[0]
        self.var_size = var_dict
        self.X = X
        self.index = index
        self.hom = hom
        self.status = 0

        self.X_var = cp.Variable((self.X_dim, self.X_dim), PSD=True)
        self.X_new = None
        self.z_new = None
        self.z_prev = None

    @staticmethod
    def create_admm_cliques_from_problem(problem: HomQCQP, variable=["x_", "z_"]):
        """
        Generate the cliques to be used in ADMM.

        The generated F and G matrices are the vectorized versions of the consistency constraints.
        They are such that F @ this.flatten() + G @ other.flatten() == 0, where other is the stack
        of all flattened cliques at the indices given in variables_FG.
        """
        from cert_tools.base_clique import get_chain_clique_data

        clique_data = get_chain_clique_data(
            problem.var_sizes, fixed=["h"], variable=variable
        )
        problem.clique_decomposition(clique_data=clique_data)

        eq_list = problem.get_consistency_constraints()

        Q_dict = problem.decompose_matrix(problem.C, method="split")
        A_dict_list = [problem.decompose_matrix(A, method="first") for A in problem.As]
        admm_cliques = []
        for clique in problem.cliques:
            Constraints = [problem.get_homog_constraint(clique.var_sizes)]
            for A_dict in A_dict_list:
                if clique.index in A_dict.keys():
                    Constraints.append(
                        (A_dict[clique.index].get_matrix(clique.var_sizes), 0.0)
                    )
            admm_clique = ADMMClique(
                Q=Q_dict[clique.index].get_matrix(clique.var_sizes),
                Constraints=Constraints,
                var_dict=clique.var_sizes,
                index=clique.index,
                hom="h",
            )

            # find all overlapping constraints involving this clique
            # <Ak, Xk> + <Al, Xl> = 0
            # F @ vech(Xk) + G @ vech(Xl) = 0
            F_dict = dict()
            G_dict = dict()
            for k, l, Ak, Al in eq_list:
                if k == clique.index:
                    if l in F_dict:
                        # TODO(FD) we currently need to use the full matrix and not just the upper half
                        # because it's not trivial to extract the upper half of a matrix in the Fusion API.
                        # We might change that later.
                        F_dict[l] = sp.vstack([F_dict[l], Ak.reshape(1, -1)])
                        G_dict[l] = sp.vstack([G_dict[l], Al.reshape(1, -1)])
                    else:
                        F_dict[l] = Ak.reshape(1, -1)
                        G_dict[l] = Al.reshape(1, -1)
                # TODO(FD) I am not 100% this is needed. For dSDP this would be counting constraints double.
                # But it seems like for ADMM it is required, because for example, the first clique in a chain
                # would otherwise not be linked to any other cliques through consensus constraints.
                elif l == clique.index:
                    if k in F_dict:
                        F_dict[k] = sp.vstack([F_dict[k], Al.reshape(1, -1)])
                        G_dict[k] = sp.vstack([G_dict[k], Ak.reshape(1, -1)])
                    else:
                        F_dict[k] = Al.reshape(1, -1)
                        G_dict[k] = Ak.reshape(1, -1)

            admm_clique.F = sp.vstack(F_dict.values()) if len(F_dict) else None
            admm_clique.G_dict = G_dict
            admm_clique.sigmas = (
                np.zeros(admm_clique.F.shape[0]) if len(F_dict) else None
            )
            admm_cliques.append(admm_clique)
        return admm_cliques

    def current_error(self):
        return np.sum(np.abs(self.F @ self.X_new - self.z_new))

    def get_constraints_cvxpy(self, X):
        return [cp.trace(A_k @ X) == b_k for A_k, b_k in zip(self.Constraints)]

    def get_objective_cvxpy(self, X, rho_k, adjust=False, scale_method="fro"):
        if adjust:
            Q_here, scale, offset = adjust_Q(self.Q, scale_method=scale_method)
        else:
            Q_here = self.Q
            scale = 1.0
            offset = 0.0
        if np.ndim(rho_k) > 0:
            objective = cp.Minimize(
                cp.trace(Q_here @ X)
                + self.sigmas.T @ (self.F @ X.flatten() - self.g)
                + 0.5
                * cp.norm2(cp.multiply(self.F @ X.flatten() - self.g, np.sqrt(rho_k)))
                ** 2
            )
        else:
            objective = cp.Minimize(
                cp.trace(Q_here @ X)
                + self.sigmas.T @ (self.F @ X.flatten() - self.g)
                + 0.5 * rho_k * cp.norm2(self.F @ X.flatten() - self.g) ** 2
            )
        return objective, scale, offset

    def update_rho(c, mu_rho, tau_rho, individual_rho):
        if np.ndim(c.rho_k) > 0:
            c.rho_k = update_rho(
                c.rho_k, c.dual_res_k, c.primal_res_k, mu=mu_rho, tau=tau_rho
            )
        else:
            c.rho_k = update_rho(
                c.rho_k,
                np.linalg.norm(c.dual_res_k),
                np.linalg.norm(c.primal_res_k),
                mu=mu_rho,
                tau=tau_rho,
            )
