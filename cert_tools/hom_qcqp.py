import warnings
from time import time

import chompack
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from cvxopt import amd, spmatrix
from igraph import Graph
from igraph import plot as plot_graph
from poly_matrix import PolyMatrix

from cert_tools.base_clique import BaseClique
from cert_tools.linalg_tools import find_dependent_columns, smat, svec


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

    def __init__(self, homog_var="h"):
        # Define variable dictionary.
        # keys: variable names used in cost and constraint PolyMatrix definitions.
        # values: the size of each variable
        self.var_sizes = dict(h=1)  # dictionary of sizes of variables
        self.var_inds = None  # dictionary of starting indices of variables
        self.var_list = []  # List of variables
        self.dim = 0  # total size of variables
        self.C = None  # cost matrix
        self.As = None  # list of constraints
        self.asg = Graph()  # Aggregate sparsity graph
        self.cliques = []  # List of clique objects
        self.order = []  # Elimination ordering
        self.var_clique_map = {}  # Variable to clique mapping (maps to set)
        self.h = homog_var  # Homogenizing variable name

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

    def remove_dependent_constraints(self):
        """Remove dependent constraints from the list of constraints"""

        # Get the matrix representation of the constraints
        As = [svec(A.get_matrix(self.var_sizes)) for A in self.As]
        # Get the rank of the constraints
        A = sp.vstack(As).T
        # Find dependent columns
        bad_idx = find_dependent_columns(A, tolerance=1e-10)
        # Remove dependent constraints
        self.As = [self.As[i] for i in range(len(self.As)) if i not in bad_idx]

    def get_asg(self, rm_homog=False):
        """Generate Aggregate Sparsity Graph for a given problem. Note that values
        of the matrices are irrelevant.

        Args:
            pmats (list[PolyMatrix]): _description_
            variables (_type_, optional): _description_. Defaults to None.

        Returns:
            ig.Graph: _description_
        """

        # Combine cost and constraints
        # NOTE: The numerical values here do not matter, just whether or not an element of the matrix is filled.
        pmat = self.C
        for A in self.As:
            pmat += A
        # build variable dictionaries
        self.var_sizes = pmat.variable_dict_i.copy()
        self._update_variables()

        # generate edges and vertices
        edge_list = []
        variables = set()
        # ensure symmetric
        assert pmat.symmetric, ValueError("Input PolyMatrices must be symmetric")
        # Loop through filled matrix elements
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
            G.delete_vertices(G.vs.select(name_eq=self.h))
        # Remove any self loops or multi-edges
        G.simplify(loops=True, multiple=True)
        # Store the sparsity graph
        self.asg = G

    @staticmethod
    def merge_cosmo(cp, ck, np, nk):
        """clique merge function from COSMO paper:
        https://arxiv.org/pdf/1901.10887

        Uses the metric:
        Cp^3  + Ck^3 - (Cp U Ck)^3 > 0

        Args:
            cp (int): clique order of parent
            ck (int): clique order of child
            np (int): supernode order of parent
            nk (int): supernode order of child
        """
        # Metric: Cp^3  + Ck^3 - (Cp + Nk)^3
        return cp**3 + ck**3 > (cp + nk) ** 3

    def clique_decomposition(self, elim_order="amd", merge_function=None):
        """Uses CHOMPACK to get the maximal cliques and build the clique tree
        The clique objects are stored in a list. Each clique object stores information
        about its parents and children, as well as separators"""
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
        self.symb = chompack.symbolic(S, p=p, merge_function=merge_function)
        # NOTE: reordered set to false so that everything is expressed in terms of the original variables.
        cliques = self.symb.cliques()
        sepsets = self.symb.separators()
        parents = self.symb.parent()
        residuals = self.symb.supernodes()
        var_list_perm = [self.asg.vs["name"][p] for p in self.symb.p]
        # Define clique objects
        for i, clique in enumerate(cliques):
            # separator as variables
            sepset_vars = [var_list_perm[v] for v in sepsets[i]]
            # Order the clique variable list so that the separators are first
            # NOTE: This is used for minimum rank completion
            varlist = sepsets[i] + residuals[i]
            # Store variable information for each clique
            clique_var_sizes = {}  # dict for storing variable sizes in a clique
            for v in varlist:
                varname = var_list_perm[v]
                clique_var_sizes[varname] = self.var_sizes[varname]
                # Update map between variables and cliques
                if varname not in self.var_clique_map.keys():
                    self.var_clique_map[varname] = set()
                self.var_clique_map[varname].add(i)

            # Define clique object and add to list
            clique_obj = BaseClique(
                index=i,
                var_sizes=clique_var_sizes,
                separator=sepset_vars,
                parent=parents[i],
            )
            self.cliques.append(clique_obj)

    def get_problem_matrices(self):
        """Get sparse, numerical form of objective and constraint matrices
        for use in optimization"""
        # convert cost to sparse matrix
        cost = self.C.get_matrix(self.var_sizes)
        # Define other constraints
        constraints = [(A.get_matrix(self.var_sizes), 0.0) for A in self.As]
        # define homogenizing constraint
        Ah = PolyMatrix()
        Ah[self.h, self.h] = 1
        homog_constraint = (Ah.get_matrix(self.var_sizes), 1.0)
        constraints.append(homog_constraint)
        return cost, constraints

    def get_standard_form(self, vec_order="C"):
        """Returns the problem in standard form for input to solvers like SCS, Clarabel, COSMO, SparseCoLo, etc.
        Note that the Hom QCQP is in the standard dual conic form:
        max    -x^T P x - b^T z
        s.t.    P*x + A^T z = -q
                z \in PSD
        Args:
            vec_order (str, optional): Order of vectorization on triu indices. Defaults to "C". Clarabel uses "C", SCS uses "R"
        """
        # Retrieve problem matrices
        C, constraints = self.get_problem_matrices()
        # get constraints
        A = []
        b = []
        for A_mat, b_val in constraints:
            A.append(svec(A_mat, vec_order))
            b.append(b_val)
        A = sp.vstack(A).T
        A = sp.csc_matrix(A)
        q = -np.array(b)
        # quadratic part (zeros)
        P = sp.csc_matrix((len(q), len(q)))
        # get cost
        b = svec(C.toarray(), vec_order)
        return P, q, A, b

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
            # Get parent clique object and separator set
            k = clique_l.parent
            clique_k = self.cliques[k]
            sepset = clique_l.separator
            # loop through variable combinations
            for i, var1 in enumerate(sepset):
                for var2 in sepset[i:]:
                    # loop through rows and columns
                    for row in range(self.var_sizes[var1]):
                        for col in range(self.var_sizes[var2]):
                            if var1 == var2:
                                if col < row:
                                    # skip lower diagonal
                                    continue
                                elif col > row:
                                    # Make mat symmetric
                                    vals = [1.0, 1.0]
                                    rows = [row, col]
                                    cols = [col, row]

                                else:
                                    vals = [1.0]
                                    rows = [row]
                                    cols = [col]
                            else:
                                vals = [1.0]
                                rows = [row]
                                cols = [col]
                            mat = sp.coo_matrix(
                                (vals, (rows, cols)),
                                (self.var_sizes[var1], self.var_sizes[var2]),
                            )
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

    def plot_asg(self, remove_vars=[], block=True, plot_fill=True):
        """plot aggregate sparsity pattern

        Args:
            G (_type_, optional): _description_. Defaults to None.
        """

        G = self.asg

        # Parameters
        vertex_color = "red"
        vertex_shape = "circle"
        vertex_size = 10
        edge_width = 0.5
        edge_color = "gray"

        if plot_fill:
            G.es["fill_edge"] = [False] * len(G.es)
            A_filled = self.symb.sparsity_pattern()
            data = np.array(A_filled.V).squeeze(1)
            rows = np.array(A_filled.I).squeeze(1)
            cols = np.array(A_filled.J).squeeze(1)
            A_filled = sp.csr_matrix((data, (rows, cols)), shape=A_filled.size)
            p = np.array(self.symb.p).flatten()
            A = self.asg.get_adjacency_sparse()
            A += sp.eye(A.shape[0])
            A = A[p, :][:, p]
            ip = np.array(self.symb.ip).flatten()
            fill_in = (A_filled - A)[ip, :][:, ip]
            fill_in = sp.tril(fill_in)
            fill_in.eliminate_zeros()
            nz = fill_in.nonzero()
            edges = zip(nz[0], nz[1])
            G.add_edges(edges, attributes=dict(fill_edge=True))

            # plt.spy(A_filled, markersize=3)
            # plt.spy(A, markersize=1, color="red")
            # plt.show()

        if len(remove_vars) > 0:
            G = G.copy()
            rm_verts = G.vs.select(name_in=remove_vars)
            G.delete_vertices(rm_verts)

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
        plt.show(block=block)

    def plot_ctree(self, block=True):
        """Plot junction tree associated with the problem."""
        ctree = Graph(directed=True)
        ctree.add_vertices(len(self.cliques))
        vlabel = []
        elabel = []
        for clique in self.cliques:
            vlabel.append(f"{clique.index}:{list(clique.var_list)}")
            if len(clique.separator) > 0:
                ctree.add_edge(clique.parent, clique.index)
                elabel.append(clique.separator)
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
        plt.show(block=block)

    def get_cliques_from_psd_mat(self, mat):
        """Return clique matrices corresponding to solution matrix."""
        # If not generated, get starting indices for slicing
        if self.var_inds is None:
            self.var_inds, self.dim = self._update_variables()

        cliques = self.cliques
        clique_vars = []
        for k, clique in enumerate(cliques):
            # Get slices
            clique_vars.append(self.get_slices(mat, clique.var_list))
        return clique_vars

    @staticmethod
    def factor_psd_mat(mat, rank_tol=1e5):
        # Compute eigendecomposition in descending order
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        # Compute rank
        rankfound = False
        for r, val in enumerate(eigvals):
            if eigvals[0] / val > rank_tol:
                rankfound = True
                break
        if not rankfound:
            r = len(eigvals)

        # return factorized solution
        factor = eigvecs[:, :r] * np.sqrt(eigvals)[:r]

        return factor, r

    def get_mr_completion(self, clique_mats, rank_tol=1e5):
        """Complete a positive semidefinite completable matrix using the
        minimum-rank completion as proposed in:
        Jiang, Xin et al. “Minimum-Rank Positive Semidefinite Matrix Completion with Chordal Patterns and Applications to Semidefinite Relaxations.”

        Args:
            clique_mats (list): list of nd-arrays representing the SDP solution (per clique)
            rank_tol (_type_, optional): Tolerance for determining rank. Defaults to 1e5.
        """
        r_max = 0  # max rank found
        ranks = []
        factor_dict = {}  # dictionary of factored solution
        # traverse clique tree in inverse topological order (start at root)
        for i in range(len(self.cliques) - 1, -1, -1):
            # get corresponding clique
            clique = self.cliques[i]
            # factor solution pad if required
            factor, r = self.factor_psd_mat(clique_mats[i])
            # keep track of max rank
            if r > r_max:
                r_max = r
            ranks.append(r)
            # Pad the factor if necessary
            factor = np.pad(factor, [[0, 0], [0, r_max - factor.shape[1]]])
            # if no separator, then we are at the root, add all vars to dictionary
            if len(clique.separator) == 0:
                for key in clique.var_list:
                    inds = clique._get_indices(key)
                    factor_dict[key] = factor[inds, :]
            else:
                # divide clique solution into separator and residual
                sep_inds = clique._get_indices(clique.separator)
                res_inds = clique._get_indices(clique.residual)
                V, U = factor[sep_inds, :], factor[res_inds, :]
                # get separator from dict (should already be defined)
                Yval = np.vstack([factor_dict[var] for var in clique.separator])
                Yval = np.pad(Yval, [[0, 0], [0, r_max - Yval.shape[1]]])
                # Find the orthogonal transformation between cliques
                u1, s1, Qh1 = np.linalg.svd(Yval)
                u2, s2, Qh2 = np.linalg.svd(V)
                U_Q = U @ Qh2.T @ Qh1
                for key in clique.residual:
                    # NOTE: Assumes that separator comes before residual in variable ordering
                    inds = clique._get_indices(key) - (max(sep_inds) + 1)
                    factor_dict[key] = U_Q[inds, :]

        # Construct full factor
        Y = []
        for varname in self.var_list:
            factor = factor_dict[varname]
            Y.append(np.pad(factor, [[0, 0], [0, r_max - factor.shape[1]]]))
        Y = np.vstack(Y)
        return Y, ranks, factor_dict

    def _update_variables(self):
        "Loop through variable sizes and get starting indices"
        index = 0
        self.var_inds = {}
        for varname in self.var_sizes.keys():
            self.var_inds[varname] = index
            index += self.var_sizes[varname]
        self.dim = index
        self.var_list = list(self.var_sizes.keys())

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