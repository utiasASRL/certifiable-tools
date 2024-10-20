import warnings
from time import time

import chompack
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import sksparse.cholmod as cholmod
from cvxopt import amd, spmatrix
from igraph import Graph
from igraph import plot as plot_graph
from poly_matrix import PolyMatrix

from cert_tools.base_clique import BaseClique
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
        self.var_sizes = dict(h=1)  # dictionary of sizes of variables
        self.var_inds = None  # dictionary of starting indices of variables
        self.dim = None  # total size of variables
        self.C = None  # cost matrix
        self.As = None  # list of constraints
        self.asg = Graph()  # Aggregate sparsity graph
        self.cliques = []  # List of clique objects
        self.order = []  # Elimination ordering
        self.var_clique_map = {}  # Variable to clique mapping (maps to set)

    def define_objective(self, *args, **kwargs) -> PolyMatrix:
        """Function should define the cost matrix for the problem
        NOTE: Defined by inhereting class
        Returns:
            PolyMatrix: _description_
        """
        # Default to empty poly matrix
        return PolyMatrix()

    def define_constraints(self, *args, **kwargs) -> list[PolyMatrix]:
        """Function should define a list of PolyMatrices that represent
        the set of affine constraints for the problem.
        NOTE: Defined by inhereting class
        Returns:
            list[PolyMatrix]: _description_
        """
        return []

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

    def clique_decomposition(self, elim_order="amd"):
        """Uses CHOMPACK to get the maximal cliques and build the clique tree
        The clique objects are stored in a list. Each clique object stores information
        about its parents and children, as well as seperators"""
        if len(self.asg.vs) == 0:
            warnings.warn("Aggregate sparsity graph not defined. Building now.")
            # build aggregate sparsity graph
            self.get_asg()
        if elim_order == "amd":
            p = amd.order

        # Convert adjacency to sparsity pattern
        nvars = len(self.var_sizes)
        A = self.asg.get_adjacency_sparse() + sp.eye(nvars)
        rows, cols = A.nonzero()
        S = spmatrix(1.0, rows, cols, A.shape)
        # get information from factorization
        self.symb = chompack.symbolic(S, p=p)
        # NOTE: reordered set to false so that everything is expressed in terms of the original variables.
        cliques = self.symb.cliques()
        sepsets = self.symb.separators()
        parents = self.symb.parent()
        var_list_perm = [self.asg.vs["name"][p] for p in self.symb.p]
        # Define cliques
        for i, clique in enumerate(cliques):
            # Store variable information for each clique
            clique_var_sizes = {}  # dict for storing variable sizes in a clique
            for v in clique:
                varname = var_list_perm[v]
                clique_var_sizes[varname] = self.var_sizes[varname]
                # Update map between variables and cliques
                if varname not in self.var_clique_map.keys():
                    self.var_clique_map[varname] = set()
                self.var_clique_map[varname].add(i)
            # seperator as variables
            sepset_vars = [var_list_perm[v] for v in sepsets[i]]
            # Define clique object and add to list
            clique_obj = BaseClique(
                index=i,
                var_sizes=clique_var_sizes,
                seperator=sepset_vars,
                parent=parents[i],
            )
            self.cliques.append(clique_obj)

    def get_sdp_matrices(self):
        """Get sparse, numerical form of objective and constraint matrices
        for use in optimization"""
        # convert cost to sparse matrix
        cost = self.C.get_matrix(self.var_sizes)
        # Define other constraints
        constraints = [(A.get_matrix(self.var_sizes), 0.0) for A in self.As]
        # define homogenizing constraint
        Ah = PolyMatrix()
        Ah["h", "h"] = 1
        homog_constraint = (Ah.get_matrix(self.var_sizes), 1.0)
        constraints.append(homog_constraint)
        return cost, constraints

    def get_consistency_constraints(self):
        """Return a list of constraints that enforce equalities between
        clique variables. List consist of 4-tuples: (k, l, A_k, A_l)
        where k and l are the indices of the cliques for which the equality is
        defined and A_k and A_l are the matrices enforcing variable equality
        Equality Equation:  <A_k, X_k> + <A_l, X_l> = 0

        NOTE: The complicating factor here is that different cliques have
        different variables and orderings. We get around this by using the
        PolyMatrix module.
        """
        # Lopp through edges in the junction tree
        eq_list = []
        for l, clique_l in enumerate(self.cliques):
            # Get parent clique object and seperator set
            k = clique_l.parent
            clique_k = self.cliques[k]
            sepset = clique_l.seperator
            # loop through variable combinations
            for i, var1 in enumerate(sepset):
                for var2 in sepset[i:]:
                    # loop through rows and columns
                    for row in range(self.var_sizes[var1]):
                        for col in range(self.var_sizes[var2]):
                            # TODO This should be defined as a sparse matrix
                            mat = np.zeros((self.var_sizes[var1], self.var_sizes[var2]))
                            mat[row, col] = 1.0
                            if var1 == var2:
                                if col < row:
                                    # skip lower diagonal
                                    continue
                                elif col > row:
                                    # Make mat symmetric
                                    mat[col, row] = 1.0
                            # Define Abstract Matrix
                            A = PolyMatrix(symmetric=True)
                            A[var1, var2] = mat
                            # Get matrices, organized using clique variables
                            A_k = A.get_matrix(
                                variables=clique_k.var_sizes, output_type="coo"
                            )
                            A_l = -A.get_matrix(
                                variables=clique_l.var_sizes, output_type="coo"
                            )
                            # Ensure matrices are as sparse as possible.
                            A_k.eliminate_zeros()
                            A_l.eliminate_zeros()
                            eq_list.append((k, l, A_k, A_l))

        return eq_list

    def decompose_matrix(self, pmat: PolyMatrix, method="split"):
        """Decompose a matrix according to clique decomposition. Returns a dictionary with the key being the clique number and the value being a PolyMatrix that contains decomposed matrix on that clique."""
        assert isinstance(pmat, PolyMatrix), TypeError("Input should be a PolyMatrix")
        assert pmat.symmetric, ValueError("PolyMatrix input should be symmetric")
        dmat = {}  # defined decomposed matrix dictionary
        # Loop through elements of polymatrix
        for iVar1, var1 in enumerate(pmat.matrix.keys()):
            for var2 in pmat.matrix[var1].keys():
                # NOTE: Next line avoids double counting elements due to symmetry of matrix. There should be a better way to do this and this could be quite slow.
                if var2 in list(pmat.matrix.keys())[:iVar1]:
                    continue
                # Get the cliques that contain both variables
                cliques1 = self.var_clique_map[var1]
                cliques2 = self.var_clique_map[var2]
                cliques = cliques1 & cliques2
                # Define weighting based on method
                if method == "split":
                    alpha = np.ones(len(cliques)) / len(cliques)
                elif method == "first":
                    alpha = np.zeros(len(cliques))
                    alpha[0] = 1.0
                else:
                    raise ValueError("Decomposition method unknown")
                # Loop through cliques and add to dictionary
                for k, clique in enumerate(cliques):
                    if alpha[k] > 0.0:  # Check non-zero weighting
                        # Define polymatrix
                        pmat_k = PolyMatrix()
                        pmat_k[var1, var2] = pmat[var1, var2] * alpha[k]
                        # Add to dictionary
                        if clique not in dmat.keys():
                            dmat[clique] = pmat_k
                        else:
                            dmat[clique] += pmat_k
        return dmat

    def solve_sdp(self, method="sdp", solver="mosek", verbose=False, tol=1e-11):
        """Solve non-chordal SDP for PGO problem without using ADMM"""
        # TODO move this into the solvers module. shouldnt be here

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
            X, info = solver(
                Q=obj, Constraints=constrs, adjust=False, verbose=verbose, tol=tol
            )
        elif method == "dsdp":
            ValueError("Method not defined.")
        else:
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

    def plot_ctree(self):
        """Plot junction tree associated with the problem."""
        ctree = Graph(directed=True)
        ctree.add_vertices(len(self.cliques))
        vlabel = []
        elabel = []
        for clique in self.cliques:
            vlabel.append(f"{clique.index}:{list(clique.var_sizes.keys())}")
            if len(clique.seperator) > 0:
                ctree.add_edge(clique.parent, clique.index)
                elabel.append(clique.seperator)
            else:
                root = clique.index

        fig, ax = plt.subplots()
        plot_options = {
            "vertex_label": vlabel,
            "edge_label": elabel,
            "target": ax,
            "layout": ctree.layout_reingold_tilford(),
        }
        plot_graph(ctree, **plot_options)
        plt.title("Clique Tree")
        plt.show()

    def get_cliques_from_psd_mat(self, mat):
        """Return clique matrices corresponding to solution matrix."""
        # If not generated, get starting indices for slicing
        if self.var_inds is None:
            self.var_inds, self.dim = self._get_start_indices()

        cliques = self.cliques
        clique_vars = []
        for k, clique in enumerate(cliques):
            var_list = clique.var_sizes.keys()
            # Get slices
            clique_vars.append(self.get_slices(mat, var_list))
        return clique_vars

    def get_psdc_mat(self, clique_vars, decomp_method="split"):
        """Return positive semidefinite completable matrix derived from cliques

        Args:
            clique_vars (_type_): list of clique variable solutions
            decomp_method (str, optional): decomposition method that was used to divide cost and constraint matrices. Defaults to "split".

        Raises:
            ValueError: if decomposition method is not known

        Returns:
            lil_matrix: positive semidefinite completable matrix
        """
        # If not generated, get starting indices for slicing
        if self.var_inds is None:
            self.var_inds, self.dim = self._get_start_indices()
        # Make output matrix
        X = sp.lil_matrix((self.dim, self.dim))
        # get clique objects
        cliques = self.ctree.vs["clique"]
        # loop through variables
        varlist = list(self.var_sizes.keys())
        for i, var1 in enumerate(varlist):
            # row slice of X matrix
            x_rows = slice(
                self.var_inds[var1],
                self.var_inds[var1] + self.var_sizes[var1],
            )
            for var2 in varlist[i:]:
                # column slice of X matrix
                x_cols = slice(
                    self.var_inds[var2],
                    self.var_inds[var2] + self.var_sizes[var2],
                )
                # Find cliques that contain this variable
                clq1 = self.var_clique_map[var1]
                clq2 = self.var_clique_map[var2]
                clqs = clq1 & clq2
                if len(clqs) > 0:
                    # Define weighting based on method
                    if decomp_method == "split":
                        alpha = np.ones(len(clqs)) / len(clqs)
                    elif decomp_method == "first":
                        alpha = np.zeros(len(clqs))
                        alpha[0] = 1.0
                    else:
                        raise ValueError("Decomposition method unknown")
                    # Loop through cliques and insert weighted sums into psdc matrix
                    for k, clique_ind in enumerate(clqs):
                        if alpha[k] > 0.0:  # Check non-zero weighting
                            cvar = clique_vars[clique_ind]  # clique variable
                            X[x_rows, x_cols] += (
                                cliques[clique_ind].get_slices(cvar, [var1], [var2])
                                * alpha[k]
                            )
                            if not var1 == var2:
                                X[x_cols, x_rows] = X[x_rows, x_cols].T
        return X

    def get_psd_completion(self, cliques, method="mr"):
        """Complete a positive semidefinite completable matrix

        Args:
            psdc_mat (_type_): matrix to be completed
            method (str, optional): method used to complete. Defaults to "mr".
        """
        # get inverse topological ordering for the junction tree
        order = self.ctree.topological_sorting(mode="in")
        return None

    def _get_start_indices(self):
        "Loop through variable sizes and get starting indices"
        index = 0
        var_inds = {}
        for varname in self.var_sizes.keys():
            var_inds[varname] = index
            index += self.var_sizes[varname]
        size = index
        return var_inds, size

    def get_slices(self, mat, var_list_row, var_list_col=[]):
        """Get slices according to prescribed variable ordering.
        If one list provided then slices are assumed to be symmetric. If two lists are provided, they are interpreted as the row and column lists, respectively.
        """
        slices = []
        # Get index slices for the rows
        for varname in var_list_row:
            start = self.var_inds[varname]
            end = self.var_inds[varname] + self.var_sizes[varname]
            slices.append(np.array(range(start, end)))
        inds1 = np.hstack(slices)
        # Get index slices for the columns
        if len(var_list_col) > 0:
            for varname in var_list_col:
                start = self.var_inds[varname]
                end = self.var_inds[varname] + self.var_sizes[varname]
                slices.append(np.array(range(start, end)))
            inds2 = np.hstack(slices)
        else:
            # If not defined use the same list as rows
            inds2 = inds1

        return mat[np.ix_(inds1, inds2)]


def get_pattern(G: Graph):
    """Get a matrix representing the sparsity pattern of a graph"""
    # Get sparsity
    A = G.get_adjacency_sparse()
    pattern = sp.eye(len(G.vs)) + A
    pattern = abs(pattern.sign())

    return pattern
