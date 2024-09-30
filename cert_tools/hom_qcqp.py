from time import time

import matplotlib.pyplot as plt
import scipy.sparse as sp
import sksparse.cholmod as cholmod
from igraph import Graph
from igraph import plot as plot_graph
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
        # Aggregate sparsity graph
        self.asg = Graph()
        # Junction tree
        self.jtree = Graph()
        # Elimination ordering
        self.order = []

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

    def get_asg(self, rm_homog=False):
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
        # Remove any self loops or multi-edges
        G.simplify(loops=True, multiple=True)
        # Store the sparsity graph
        self.asg = G

    def build_jtree(self):
        """Build the junction tree associated with the aggregate sparsity
        graph for this problem. It is assumed that the graph has been
        triangulated (i.e, it is chordal).

        NOTE: It is likely that this could be made more efficient by combining
        it with the the graph triangulation step.
        """
        # Find maximal cliques
        cliques = self.asg.maximal_cliques()
        # Define the junction graph
        junction = Graph()
        junction.add_vertices(len(cliques))
        junction.vs["vlist"] = [
            [self.asg.vs["name"][node] for node in clique] for clique in cliques
        ]
        # Build edges in junction tree based on clique overlap
        clique_obj = []
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                # Get seperator set for list (intersection of cliques)
                sepset = set(cliques[i]) & set(cliques[j])
                if len(sepset) > 0:
                    # Convert to variable names
                    sepset_vars = [self.asg.vs["name"][v] for v in sepset]
                    # Create edge in junction graph
                    junction.add_edge(i, j, weight=-len(sepset), sepset=sepset_vars)
        # Get Junction tree
        self.jtree = junction.spanning_tree(
            weights=junction.es["weight"], return_tree=True
        )

    def get_sdp_matrices(self):
        """Get sparse, numerical form of objective and constraint matrices
        for use in optimization"""
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

    def solve_sdp(self, method="sdp", solver="mosek", verbose=False):
        """Solve non-chordal SDP for PGO problem without using ADMM"""
        # Get matrices
        obj, constrs = self.get_sdp_matrices()
        # Select the solver
        if solver == "mosek":
            solver = solve_sdp_mosek
        else:
            raise ValueError("Solver not supported")
        # Solve SDP
        start_time = time()
        if method == "sdp":
            X, info = solver(Q=obj, Constraints=constrs, adjust=False, verbose=verbose)
        elif method == "dsdp":

            ValueError("Method not defined.")
        # Get solution time.
        solve_time = time() - start_time

        return X, info, solve_time

    def get_elim_order(self, method="amd"):
        """Get elimination ordering for the problem

        Args:
            method (str, optional): _description_. Defaults to "amd".

        Returns:
            _type_: _description_
        """
        if self.asg is None:
            raise ValueError(
                "Aggregate sparsity graph must be defined prior to attaining elimination ordering"
            )
        pattern = get_pattern(self.asg)
        factor = cholmod.analyze(pattern, mode="simplicial", ordering_method=method)
        self.order = factor.P()

    def plot_asg(self, G=None):
        """plot aggregate sparsity pattern

        Args:
            G (_type_, optional): _description_. Defaults to None.
        """
        if G is None:
            G = self.asg

        # Parameters
        vertex_color = "red"
        vertex_shape = "circle"
        vertex_size = 10
        edge_width = 0.5
        edge_color = "gray"

        if "fill_edge" not in self.asg.es.attribute_names():
            G.es["fill_edge"] = [False] * len(G.es)
        edge_color = []
        for fill_edge in G.es["fill_edge"]:
            if fill_edge:
                edge_color.append("blue")
            else:
                edge_color.append("gray")

        # Plot
        fig, ax = plt.subplots()
        plot_graph(
            G,
            target=ax,
            vertex_size=vertex_size,
            vertex_color=vertex_color,
            vertex_shape=vertex_shape,
            vertex_label=G.vs["name"],
            edge_width=edge_width,
            edge_color=edge_color,
            margin=20,
        )
        plt.show()

    def plot_jtree(self):
        """Plot junction tree associated with the problem."""
        junction_tree = self.jtree
        fig, ax = plt.subplots()
        # relabel cliques
        plot_options = {
            "vertex_label": junction_tree.vs["vlist"],
            "edge_label": junction_tree.es["sepset"],
            "target": ax,
        }
        plot_graph(junction_tree, **plot_options)
        plt.title("Junction Tree")
        plt.show()

    def triangulate_graph(self, elim_method="amd"):
        """Find an elimination ordering.
        Loop through elimination ordering and add fill in edges.
        Adds an edge attribute to the graph that specifies if an edge is a fill in
        edge."""
        G = self.asg
        self.get_elim_order(method=elim_method)

        # Define attribute to track whether an edge is a fill edge
        G.es.set_attribute_values("fill_edge", [False] * len(G.es))
        # Define attribute to keep track of eliminated nodes
        G.vs.set_attribute_values("elim", [False] * len(G.vs))

        # init fill data
        fill_total = 0
        # Get list of vertices to eliminate
        vert_order = [G.vs[i] for i in self.order]
        for v in vert_order:
            # Get Neighborhood
            N = G.vs[G.neighbors(v)]
            for i, n1 in enumerate(N):
                if n1["elim"]:  # skip if eliminated already
                    continue
                for n2 in N[(i + 1) :]:
                    if n2["elim"] or n1 == n2:  # skip if eliminated already
                        continue
                    if not G.are_connected(n1, n2):
                        # Add a fill in edge
                        attr_dict = dict(
                            name=f"f{fill_total}",
                            fill_edge=True,
                        )
                        G.add_edge(n1, n2, **attr_dict)
                        fill_total += 1
            # Mark vertex as eliminated
            v["elim"] = True
        return fill_total


def get_pattern(G: Graph):
    """Get a matrix representing the sparsity pattern of a graph"""
    # Get sparsity
    A = G.get_adjacency_sparse()
    pattern = sp.eye(len(G.vs)) + A
    pattern = abs(pattern.sign())

    return pattern
