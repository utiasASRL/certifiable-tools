from time import time

from igraph import Graph
from poly_matrix import PolyMatrix

from cert_tools.sdp_solvers import solve_sdp_mosek


class HomQCQP:
    """Abstract class used to represent a the SDP relaxation of a
    non-convex homogenized, quadratically constrainted quadratic problem
    The general form of the problem is as follows:
     min    <C, X>
     X
        s.t. <A_i, X> = 0, forall i=1...m
             <A_0, X> = 1,
             X == PSD(n)
    """

    def __init__(self):
        # Define variable dictionary.
        # keys: variable names used in cost and constraint PolyMatrix definitions.
        # values: the size of each variable
        self.var_dict = dict("h", 1)
        # Define cost matrix
        self.def_cost_matrix()
        # Define list of constraints
        self.def_constraint_list()

    def define_objective(self, *args, **kwargs) -> PolyMatrix:
        """Function should define the cost matrix for the problem

        Returns:
            PolyMatrix: _description_
        """
        # Default to empty poly matrix
        self.C = PolyMatrix()

    def define_constraints(self, *args, **kwargs) -> list[PolyMatrix]:
        """Function should define a list of PolyMatrices that represent
        the set of affine constraints for the problem.

        Returns:
            list[PolyMatrix]: _description_
        """
        self.As = []

    def get_aggspar_graph(self, rm_homog=False):
        """Generate Aggregate Sparsity Graph for a given problem. Note that values
        of the matrices are irrelevant.

        Args:
            pmats (list[PolyMatrix]): _description_
            variables (_type_, optional): _description_. Defaults to None.

        Returns:
            ig.Graph: _description_
        """

        # Add edges corresponding to aggregate sparsity
        pmats = [self.C] + self.As
        edge_list = []
        variables = set()
        for i, pmat in enumerate(pmats):
            # ensure symmetric
            assert pmat.symmetric, ValueError("Input PolyMatrices must be symmetric")
            for key1 in pmat.matrix.keys():
                for key2 in pmat.matrix[key1]:
                    # Add to set of variables
                    variables.add(key1)
                    variables.add(key2)
                    if key1 is not key2:
                        edge_list.append((key1, key2))

        G = Graph(directed=False)
        # Create vertices based on variables
        G.add_vertices(len(variables))
        G.vs["name"] = list(variables)
        # Add edges
        G.add_edges(edge_list)
        # Remove homogenizing variable from aggregate sparsity graph
        if rm_homog:
            G.delete_vertices(G.vs.select(name_eq="h"))
        # Store the sparsity graph
        self.G_agg = G

    def get_sdp_matrices(self):
        """Get sparse, numerical form of objective and constraint matrices"""
        # convert cost to sparse matrix
        cost = self.C.get_matrix(self.var_dict)
        # Define other constraints
        constraints = [(A.get_matrix(self.var_dict), 0.0) for A in self.As]
        # define homogenizing constraint
        Ah = PolyMatrix()
        Ah["h", "h"] = 1
        homog_constraint = (Ah.get_matrix(self.var_dict), 1.0)
        constraints.append(homog_constraint)
        return cost, constraints

    def solve_sdp(self, method="mosek", verbose=False):
        """Solve non-chordal SDP for PGO problem without using ADMM"""
        # Get matrices
        obj, constrs = self.get_sdp_matrices()
        # Solve non-Homogenized SDP
        start_time = time()
        if method == "mosek":
            X, info = solve_sdp_mosek(
                Q=obj, Constraints=constrs, adjust=False, verbose=verbose
            )
        else:
            ValueError("Method not defined.")
        # Get solution time.
        solve_time = time() - start_time

        return X, info, solve_time
