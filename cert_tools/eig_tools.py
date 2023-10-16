# Maths
import numpy as np
import scipy.sparse as sp

def get_min_eigpairs(H, method="lanczos", k=6, tol=1e-8, v0=None,**kwargs):
    """Wrapper function for calling different minimum eigenvalue methods"""
    if method == "direct":
        if sp.issparse(H):
            H = H.todense()
        eig_vals, eig_vecs = np.linalg.eigh(H)
    elif method == "lanczos":
        if not sp.issparse(H):
            H = sp.csr_array(H)
        eig_vals, eig_vecs = sp.linalg.eigsh(
            H, k=k, which="SA", return_eigenvectors=True,v0=v0
        )
    elif method == "lanczos-shifted":
        if not sp.issparse(H):
            H = sp.csr_array(H)
        eig_vals, eig_vecs = min_eigs_lanczos(H, k=k, tol=tol, v0=v0, **kwargs)
    elif method == "precond-lanczos":
        pass
    elif method == "lobpcg":
        if not sp.issparse(H):
            H = sp.csr_array(H)
        eig_vals, eig_vecs = sp.linalg.lobpcg(H,X=v0,largest=False)
    else:
        raise ValueError(f"method {method} not recognized.")

    # Get min eigpairs
    sortind = np.argsort(eig_vals)
    eig_vals = eig_vals[sortind[:k]]
    eig_vecs = eig_vecs[:, sortind[:k]]
    return eig_vals, eig_vecs


def min_eigs_lanczos(H, k=6, tol=1e-6, v0=None, **kwargs):
    """Use the Lanczos process to get an approximation of minimum eigenpairs.
    For now just returning only one pair, even if the eigenspace has dimension > 1
    TODO: Address higher dimensional min eigenspace.
    """
    # Compute Coarse Max Eig
    eig_opts = dict(k=k, which="LM", return_eigenvectors=True, v0=v0)
    vals, V = sp.linalg.eigsh(H, tol=1e-3, **eig_opts)
    max_eig = np.max(vals)
    if max_eig > -tol:
        # Shift the certificate matrix by 2 lambda max. This improves
        # conditioning of the matrix.
        H_shift = H - 2 * sp.eye(H.shape[0]) * max_eig
        # try to converge to the requested number of eigenpairs
        try:
            eig_vals, eig_vecs = sp.linalg.eigsh(H_shift, **eig_opts)
        except sp.linalg.ArpackNoConvergence as err:
            # Retrieve converged values
            print(err)
            print('Retrieving converged eigenpairs')
            eig_vals = err.eigenvalues
            eig_vecs = err.eigenvectors
        eig_vals = eig_vals + 2 * max_eig
    else:
        # Largest eigenvalue is already negative. Rerun with lower tolerance
        eig_vals, eig_vecs = sp.linalg.eigsh(H, **eig_opts)

    return eig_vals, eig_vecs
