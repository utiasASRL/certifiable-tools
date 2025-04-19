import warnings
from collections import deque

import chompack
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy.sparse as sp
from cvxopt import amd, spmatrix
from igraph import Graph
from poly_matrix import PolyMatrix
from scipy.linalg import polar

from cert_tools.base_clique import BaseClique
from cert_tools.linalg_tools import find_dependent_columns, svec


def infer_var_list(clique_data):
    """Infer the variable list from clique data to fix the order."""
    if isinstance(clique_data, list):
        var_list = []
        var_list_quick_lookup = {}
        for c in clique_data:
            for ci in c:
                if ci not in var_list_quick_lookup:
                    var_list.append(ci)
                    var_list_quick_lookup[ci] = 1
        return var_list
    else:
        return None


class HomQCQP(object):
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
        self.As = None  # list of equality constraints
        self.Bs = []  # list of inequality constraints
        self.asg = Graph()  # Aggregate sparsity graph
        self.cliques = []  # List of clique objects
        self.order = []  # Elimination ordering
        self.var_clique_map = {}  # Variable to clique mapping (maps to set)
        self.h = homog_var  # Homogenizing variable name

    @staticmethod
    def create_from_matrices(C, A_list, B_list, homog_var="h"):
        homQCQP = HomQCQP(homog_var=homog_var)
        homQCQP.C = C
        homQCQP.As = A_list
        homQCQP.Bs = B_list
        return homQCQP

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
        if len(self.asg.vs) == 0:
            warnings.warn("Aggregate sparsity graph not defined. Building now.")
            # build aggregate sparsity graph
            self.get_asg()
        # Get the matrix representation of the constraints
        As = [svec(A.get_matrix(self.var_sizes)) for A in self.As]
        # Get the rank of the constraints
        A = sp.vstack(As).T
        # Find dependent columns
        bad_idx = find_dependent_columns(A, tolerance=1e-10)
        # Remove dependent constraints
        self.As = [self.As[i] for i in range(len(self.As)) if i not in bad_idx]

        return bad_idx

    def get_asg(self, var_list=None, rm_homog=False):
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
        pmat = self.C.copy()
        for A in self.As:
            pmat += A
        # build variable dictionaries
        var_sizes = pmat.variable_dict_i.copy()
        # If list is specified then align dictionary to list
        if var_list is not None:
            self.var_sizes = {key: var_sizes[key] for key in var_list}
        else:
            self.var_sizes = var_sizes
        # Update indices
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

    def clique_decomposition(
        self, elim_order="amd", clique_data=[], merge_function=None
    ):
        """Uses CHOMPACK to get the maximal cliques and build the clique tree
        The clique objects are stored in a list. Each clique object stores information
        about its parents and children, as well as separators"""
        if len(self.asg.vs) == 0:
            warnings.warn("Aggregate sparsity graph not defined. Building now.")
            # build aggregate sparsity graph
            var_list = None
            if len(clique_data):
                var_list = infer_var_list(clique_data)
            self.get_asg(var_list=var_list)

        if len(clique_data) == 0:
            if elim_order == "amd":
                p = amd.order
            # Convert adjacency to sparsity pattern
            nvars = len(self.var_sizes)
            A = self.asg.get_adjacency_sparse() + sp.eye(nvars)
            rows, cols = A.nonzero()
            S = spmatrix(1.0, rows, cols, A.shape)
            # get information from factorization
            self.symb = chompack.symbolic(S, p=p, merge_function=merge_function)
            cliques = self.symb.cliques()
            sepsets = self.symb.separators()
            parents = self.symb.parent()
            # Get variable list in permuted order (symbolic factorization reorders things)
            var_list_perm = [self.asg.vs["name"][p] for p in self.symb.p]
            # Convert indices into labels
            for i, clique in enumerate(cliques):
                cliques[i] = set([var_list_perm[v] for v in clique])
                sepsets[i] = set([var_list_perm[v] for v in sepsets[i]])
        else:
            # Get information from input cliques
            cliques, sepsets, parents = HomQCQP.process_clique_data(clique_data)

        # Reset stored clique data
        self.var_clique_map = {}
        self.cliques = []
        # Define clique objects
        for i, clique in enumerate(cliques):
            # Store variable information for each clique
            clique_var_sizes = {}  # dict for storing variable sizes in a clique
            for varname in clique:
                clique_var_sizes[varname] = self.var_sizes[varname]
                # Update map between variables and cliques
                if varname not in self.var_clique_map.keys():
                    self.var_clique_map[varname] = set()
                self.var_clique_map[varname].add(i)

            # Define clique object and add to list
            clique_obj = BaseClique(
                index=i,
                var_sizes=clique_var_sizes,
                separator=set(sepsets[i]),
                parent=parents[i],
            )
            self.cliques.append(clique_obj)

    @staticmethod
    def process_clique_data(clique_data):
        """Process clique data. If the input data is a dictionary, it is assumed that the "cliques", "separators", and "parents" are defined as keys, each providing list. Ordering of these lists should be topological, meaning that any parent should be ordered after their child. "cliques" contains a list of sets of clique variables, "parents" indicates the index of the parent in the clique list, and "separators" is a list of the separator (intersection) between each clique and its parent.
        Otherwise the elements of the list must be sets containing the variable names of the variables in a given clique. In this case, a clique tree is built using a minimum spanning tree of the clique graph.

        Args:
            clique_data (list): list of collections of variable names representing cliques. Use a list for each clique group if you care about the order. Use a set if the order does not matter.
            clique_data (dict): a dictionary of the clique data with keys
        """
        if isinstance(clique_data, dict):
            clique_list = clique_data["cliques"]
            separators = clique_data["separators"]
            parents = clique_data["parents"]
            # Check that parents are defined properly
            for child_ind, parent_ind in enumerate(parents):
                assert child_ind <= parent_ind, ValueError(
                    f"Topological ordering violated: parent index {parent_ind} has lower order than child index {child_ind}"
                )
        elif isinstance(clique_data, list):
            # build clique tree from clique list
            cliques = clique_data
            edges, sepsets, weights = [], [], []
            for v1 in range(len(cliques) - 1):
                for v2 in range(v1 + 1, len(cliques)):
                    if isinstance(cliques[v1], set):
                        sepset = cliques[v1] & cliques[v2]
                    else:
                        sepset = [c for c in cliques[v2] if c in cliques[v1]]
                    weight = len(sepset)
                    if weight > 0:
                        edges.append((v1, v2))
                        weights.append(-weight)
                        sepsets.append(sepset)
            prcsd = [False] * len(weights)
            # Create clique graph
            ctree = Graph()
            ctree.add_vertices(len(cliques))
            ctree.add_edges(
                edges, attributes={"weight": weights, "sepset": sepsets, "prcsd": prcsd}
            )
            # Get clique tree
            ctree = ctree.spanning_tree(weights=ctree.es["weight"], return_tree=True)
            # Process the cliques using a fifo queue
            clique_queue = deque()  # Faster than a list for fifo queue
            # add root to lists
            root_id = len(cliques) - 1  # Rooted at final clique
            clique_queue.appendleft(root_id)  # add root node to queue
            clique_list = [cliques[root_id]]  # add root clique
            separators = [{}]  # root separator is empty
            parent_ids = [root_id]  # root parent is self.
            clique_ids = [root_id]  # id
            while len(clique_queue) > 0:
                parent_id = clique_queue.pop()
                # process children
                child_ids = ctree.neighbors(parent_id)
                for child_id in child_ids:
                    # Get associated edge and mark as processed
                    edge = ctree.es.select(_within=[parent_id, child_id])[0]
                    clique_list.append(cliques[child_id])
                    separators.append(edge["sepset"])
                    parent_ids.append(parent_id)  # Store index of parent in list
                    clique_ids.append(child_id)
                    # Add child to queue to be processed
                    clique_queue.appendleft(child_id)
                    # Remove edge from tree so that we don't reprocess the edge
                    edge.delete()

            # Flip the order of the list so that children appear before parents, parent indices must also be inverted
            clique_list = clique_list[::-1]
            separators = separators[::-1]
            parent_ids = parent_ids[::-1]
            clique_ids = clique_ids[::-1]
            # retrieve indices of parents in list
            parents = []
            for i, parent_id in enumerate(parent_ids):
                # this search should be short since we start at i
                for j in range(i, len(clique_ids)):
                    if parent_id == clique_ids[j]:
                        parents.append(j)
                        break

        else:
            raise ValueError("Clique data must be dictionary or list.")

        return clique_list, separators, parents

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

        # for backward compatibility, only returning B_list if needed.
        if len(self.Bs):
            B_list = [B.get_matrix(self.var_sizes) for B in self.Bs]
            return cost, constraints, B_list
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
            if len(sepset) == 0:
                continue
            # Get indices for each clique on separator
            indices_k = clique_k._get_indices(var_list=sepset)
            indices_l = clique_l._get_indices(var_list=sepset)
            size_k = clique_k.size
            size_l = clique_l.size
            # Define sparse constraint matrices for each element in the seperator overlap
            for i in range(len(indices_k)):
                for j in range(i, len(indices_k)):
                    if i == j:
                        vals_k = [1.0]
                        rows_k = [indices_k[i]]
                        cols_k = [indices_k[j]]
                        vals_l = [-1.0]
                        rows_l = [indices_l[i]]
                        cols_l = [indices_l[j]]
                    else:
                        vals_k = [1.0, 1.0]
                        rows_k = [indices_k[i], indices_k[j]]
                        cols_k = [indices_k[j], indices_k[i]]
                        vals_l = [-1.0, -1.0]
                        rows_l = [indices_l[i], indices_l[j]]
                        cols_l = [indices_l[j], indices_l[i]]
                    A_k = sp.coo_matrix(
                        (vals_k, (rows_k, cols_k)),
                        (size_k, size_k),
                    )
                    A_l = sp.coo_matrix(
                        (vals_l, (rows_l, cols_l)),
                        (size_l, size_l),
                    )
                    eq_list.append((k, l, A_k, A_l))

        return eq_list

    def decompose_matrix(self, pmat: PolyMatrix, method="split"):
        """Decompose a matrix according to clique decomposition. Returns a dictionary with the key being the clique number and the value being a PolyMatrix that contains decomposed matrix on that clique."""
        assert isinstance(pmat, PolyMatrix), "Input should be a PolyMatrix"
        assert pmat.symmetric, "PolyMatrix input should be symmetric"
        assert len(self.var_clique_map), "Need to first populate var_clique_map!"

        dmat = {}  # defined decomposed matrix dictionary
        # Loop through elements of polymatrix and gather information about cliques and edges
        edges = []
        clique_sets = [set() for _ in range(len(self.cliques))]
        edge_clique_map = {}
        for iVar1, var1 in enumerate(pmat.matrix.keys()):
            for var2 in pmat.matrix[var1].keys():
                # NOTE: Next line avoids double counting elements due to symmetry of matrix.
                if var2 in list(pmat.matrix.keys())[:iVar1]:
                    continue
                # Get the cliques that contain the associated edge in the ASG
                edge = (var1, var2)
                cliques = self.var_clique_map[var1] & self.var_clique_map[var2]
                edge_clique_map[edge] = cliques
                edges.append(edge)
                for clique in cliques:
                    clique_sets[clique].add(edge)

        if method == "greedy-cover":
            # Get a greedy cover for the edges
            valid_cliques = set(greedy_cover(edges, clique_sets))
        else:
            valid_cliques = set(range(len(self.cliques)))

        for edge in edges:
            cliques = edge_clique_map[edge] & valid_cliques
            # Define weighting based on method
            if method in ["split"]:
                alpha = np.ones(len(cliques)) / len(cliques)
            elif method in ["first", "greedy-cover"]:
                alpha = np.zeros(len(cliques))
                alpha[0] = 1.0
            else:
                raise ValueError("Decomposition method unknown")
            # Loop through cliques and add to dictionary
            for k, clique in enumerate(cliques):
                if alpha[k] > 0.0:  # Check non-zero weighting
                    # Define polymatrix
                    pmat_k = PolyMatrix()
                    pmat_k[edge] = pmat[edge] * alpha[k]
                    # Add to dictionary
                    if clique not in dmat.keys():
                        dmat[clique] = pmat_k
                    else:
                        dmat[clique] += pmat_k

        return dmat

    def plot_asg(self, remove_vars=[], html=None, block=True, plot_fill=True):
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
        if html:
            plot_graph(
                G,
                target=html,
                vertex_size=vertex_size,
                vertex_color=vertex_color,
                vertex_shape=vertex_shape,
                vertex_label=G.vs["name"],
                edge_width=edge_width,
                edge_color=edge_color,
                margin=20,
            )
        else:
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

    def plot_ctree(self, html=None, block=True):
        """Plot junction tree associated with the problem."""
        ctree = Graph(directed=True)
        ctree.add_vertices(len(self.cliques))
        vlabel = []
        elabel = []
        for clique in self.cliques:
            vlabel.append(f"{clique.index}:{list(clique.var_list)}")
            if len(clique.separator) > 0:
                ctree.add_edge(clique.parent, clique.index)
                elabel.append(list(clique.separator))
            else:
                root = clique.index

        plot_options = {
            "vertex_label": vlabel,
            "edge_label": elabel,
            "layout": ctree.layout_reingold_tilford(),
        }
        if html:
            plot_options["target"] = html
            plot_graph(ctree, **plot_options)
        else:
            fig, ax = plt.subplots()
            plot_options["target"] = ax
            plot_graph(ctree, **plot_options)
            ax.set_title("Clique Tree")
            plt.show(block=block)

    def get_cliques_from_sol(self, mat):
        """Return clique matrices corresponding to solution matrix."""
        cliques = self.cliques
        clique_vars = []
        for clique in cliques:
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
        factor = eigvecs[:, :r] * np.sqrt(eigvals[:r])

        return factor, r

    def get_mr_completion(self, clique_mats, var_list=None, rank_tol=1e5, debug=False):
        """Complete a positive semidefinite completable matrix using the
        minimum-rank completion as proposed in:
        Jiang, Xin et al. “Minimum-Rank Positive Semidefinite Matrix Completion with Chordal Patterns and Applications to Semidefinite Relaxations.”

        Args:
            clique_mats (list): list of nd-arrays representing the SDP solution (per clique)
            var_list (list): list of keys corresponding to the desired variable ordering. If None, then the object var_list is used
            rank_tol (_type_, optional): Tolerance for determining rank. Defaults to 1e5.
        """
        r_max = 0  # max rank found
        ranks = []
        factor_dict = {}  # dictionary of factored solution
        # traverse clique tree in inverse topological order (start at root)
        for i in range(len(self.cliques) - 1, -1, -1):
            # get corresponding clique
            clique = self.cliques[i]
            # factor solution.
            factor, r = self.factor_psd_mat(clique_mats[i], rank_tol=rank_tol)
            ranks.append(r)
            # keep track of max rank
            if r > r_max:
                r_max = r
                # pad previous factors to make dimensions consistent
                for key, val in factor_dict.items():
                    factor_dict[key] = np.pad(val, [[0, 0], [0, r_max - val.shape[1]]])
            elif r < r_max:
                # pad this factor to make its dim consistent
                factor = np.pad(factor, [[0, 0], [0, r_max - factor.shape[1]]])

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
                # Find the orthogonal transformation between cliques using the polar decomposition
                Q, H = polar(V.T @ Yval)

                if debug:
                    V_Q = V @ Q
                    U_Q = U @ Q
                    y = np.vstack([Yval, U_Q])
                    np.testing.assert_allclose(
                        y @ y.T, clique_mats[i], atol=1e-5, rtol=1e-4
                    )
                # Retrieve "rotated" residuals
                for key in clique.residual:
                    inds = clique._get_indices(key)
                    factor_dict[key] = factor[inds, :] @ Q

        # Check for and fix inverted solution
        if np.any(factor_dict[self.h] < 0):
            for key, value in factor_dict.items():
                factor_dict[key] = -value

        # Construct full factor
        if var_list is None:
            var_list = self.var_list
        Y = []
        for varname in var_list:
            Y.append(factor_dict[varname])
        Y = np.vstack(Y)
        return Y, ranks, factor_dict

    def get_dual_matrix(self, dual_cliques, var_list=None):
        """Construct the dual (certificate) matrix based on the dual variables corresponding to the cliques.

        Args:
            dual_cliques (List): List of dual clique variables to be summed into dual certificate matrix
            var_list (list, optional): List representing desired variable ordering. Warning: Using this option may result in slower runtime. Defaults to None.

        Returns:
            _type_: _description_
        """
        if var_list is None:
            # Construct using HomQCQP variable ordering
            # We can do this directly as a sparse matrix definition
            H_mat = None
            for k, clique in enumerate(self.cliques):
                clique_vars = clique.var_list
                row_inds = self._get_indices(clique_vars)
                inds = np.meshgrid(row_inds, row_inds)
                inds = tuple([ind.flatten() for ind in inds])
                vals = dual_cliques[k].reshape(-1)
                if H_mat is None:
                    H_mat = sp.csr_array((vals, inds), shape=(self.dim, self.dim))
                else:
                    H_mat += sp.csr_array((vals, inds), shape=(self.dim, self.dim))
        else:
            # To use a different variable ordering we need to use polymatrix
            H = PolyMatrix()
            for k, clique in enumerate(self.cliques):
                clique_vars = clique.var_list
                for iVar0 in range(len(clique_vars)):
                    for iVar1 in range(iVar0, len(clique_vars)):
                        var0 = clique_vars[iVar0]
                        var1 = clique_vars[iVar1]
                        inds0 = clique._get_indices(var0)
                        inds1 = clique._get_indices(var1)
                        H[var0, var1] += dual_cliques[k][np.ix_(inds0, inds1)]
            H_mat = H.get_matrix(variables=var_list)
        return H_mat

    def _update_variables(self):
        "Loop through variable sizes and get starting indices"
        index = 0
        self.var_inds = {}
        for varname in self.var_sizes.keys():
            self.var_inds[varname] = index
            index += self.var_sizes[varname]
        self.dim = index
        self.var_list = list(self.var_sizes.keys())

    def _get_indices(self, var_list):
        """get the indices corresponding to a list of variable keys

        Args:
            var_list: variable key or list of variable keys

        Returns:
            _type_: _description_
        """
        if type(var_list) is not list:
            var_list = [var_list]
        # Get index slices for the rows
        slices = []
        for varname in var_list:
            start = self.var_inds[varname]
            end = self.var_inds[varname] + self.var_sizes[varname]
            slices.append(np.array(range(start, end)))
        inds = np.hstack(slices)
        return inds

    def get_slices(self, mat, var_list_row, var_list_col=[]):
        """Get slices according to prescribed variable ordering.
        If one list provided then slices are assumed to be symmetric. If two lists are provided, they are interpreted as the row and column lists, respectively.
        """
        # Get index slices for the rows
        inds1 = self._get_indices(var_list_row)
        # Get index slices for the columns
        if len(var_list_col) > 0:
            inds2 = self._get_indices(var_list_col)
        else:
            # If not defined use the same list as rows
            inds2 = inds1

        return mat[np.ix_(inds1, inds2)]


def plot_graph(graph, **kwargs):
    layout = kwargs.get("layout", graph.layout("kk"))
    vertex_label = kwargs.get(
        "vertex_label", graph.vs["name"] if "name" in graph.vs.attributes() else None
    )
    edge_label = kwargs.get("edge_label", None)
    target = kwargs.get("target", None)
    vertex_size = kwargs.get("vertex_size", 10)
    vertex_color = kwargs.get("vertex_color", "blue")
    vertex_shape = kwargs.get("vertex_shape", "circle")
    edge_width = kwargs.get("edge_width", 1)
    edge_color = kwargs.get("edge_color", "black")
    margin = kwargs.get("margin", 20)

    if isinstance(target, plt.Axes):
        # Matplotlib plotting
        ax = target
        ig.plot(
            graph,
            target=ax,
            layout=layout,
            vertex_size=vertex_size,
            vertex_color=vertex_color,
            vertex_shape=vertex_shape,
            vertex_label=vertex_label,
            edge_width=edge_width,
            edge_color=edge_color,
            edge_label=edge_label,
            margin=margin,
        )
        plt.show()
    else:
        # Plotly plotting
        fig = go.Figure()

        # Add edges
        for idx, edge in enumerate(graph.es):
            x0, y0 = layout[edge.source]
            x1, y1 = layout[edge.target]
            if isinstance(edge_color, list):
                color = edge_color[idx]
            else:
                color = edge_color
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=edge_width, color=color),
                    hoverinfo="none",
                )
            )

        # Add vertices
        for idx, vertex in enumerate(graph.vs):
            x, y = layout[idx]
            if isinstance(vertex_color, list):
                color = vertex_color[idx]
            else:
                color = vertex_color
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=vertex_size, color=color),
                    text=vertex_label[idx] if vertex_label else "",
                    textposition="top center",
                    hoverinfo="text",
                )
            )

        # Add edge labels
        if edge_label:
            for idx, edge in enumerate(graph.es):
                x0, y0 = layout[edge.source]
                x1, y1 = layout[edge.target]
                x_mid, y_mid = (x0 + x1) / 2, (y0 + y1) / 2
                fig.add_trace(
                    go.Scatter(
                        x=[x_mid],
                        y=[y_mid],
                        mode="text",
                        text=edge_label[idx],
                        textposition="top center",
                        hoverinfo="none",
                    )
                )

        # Update layout
        fig.update_layout(
            showlegend=False,
            autosize=True,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )
        if isinstance(target, str):
            pio.write_html(fig, file=target, auto_open=False)
        else:
            fig.show()


def greedy_cover(universe, sets):
    """Solve the set cover problem using a greedy method.

    Args:
        universe (list): list of values that we want to cover
        sets (list of sets): list of sets that are used to cover the universe.

    Returns:
        list: list of indices of the sets that cover the universe
    """
    # Define uncovered set
    uncovered = set(universe)

    cover = []
    while len(uncovered) > 0:
        # Find best set
        set_ind = np.argmax([len(this_set & uncovered) for this_set in sets])
        # Add index to cover list
        cover.append(set_ind)
        # Cover the elements
        uncovered -= sets[set_ind]

    return cover


def get_pattern(G: Graph):
    """Get a matrix representing the sparsity pattern of a graph"""
    # Get sparsity
    A = G.get_adjacency_sparse()
    pattern = sp.eye(len(G.vs)) + A
    pattern = abs(pattern.sign())

    return pattern
