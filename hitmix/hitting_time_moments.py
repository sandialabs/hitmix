# Copyright 2021 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse import spdiags, coo_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import math
import warnings

def prolongation_restriction_matvec(D, A, omega_c):
    """
    LinearOperator implementing matrix-vector product of
    :math:`\Pi \, \Pi^T (D - A) \Pi \, \Pi^T` and `x`.

    Parameters
    ----------
        D: :class:`scipy.sparse.coo_matrix`
            Sparse diagonal matrix of vertex degrees.
        A: :class:`scipy.sparse.coo_matrix`
            Sparse adjacency matrix.
        omega_c: :class:`numpy.ndarray`
            Indices of complement of hitting set vertices.

    Returns
    -------
        :class:`scipy.sparse.linalg.LinearOperator`
            Linear operator implementing matrix-vector product.

    """

    # check shapes of matrices
    mA, nA = A.shape
    mD, nD = D.shape
    assert mA == nA, "Input matrix 'A' must be square"
    assert mD == nD, "Input matrix 'D' must be square"
    assert nA == nD, "Input matrices 'A' and 'D' must have the same shape"

    def matvec(x):
        # prolongation, y = \Pi * x
        y = np.zeros_like(x)
        y[omega_c] = x[omega_c]

        # matrix-vector product, y = D*(I - D^(-1)*A) * y
        y = D @ y - A @ y

        # restriction, z = \Pi^T * y
        z = np.zeros_like(y)
        z[omega_c] = y[omega_c]

        return z

    return LinearOperator((nA, nA), matvec=matvec)


def augmented_adjacency(A, omega):
    """
    Define the augmented adjacency matrix defined as the submatrix of A that do not contain
    the rows and columns of A associated with the following:
    - Vertices in the hitting set `omega`
    - Vertices with no path to vertices in the hitting set `omega` (i.e., vertices in components
    that are in components that are disconnected from those containing the hitting set vertices)
    """

    # compute connected components
    n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)

    # find the set of connected components containing the hitting set vertices
    components_of_omega = np.unique(labels[omega])

    # identify the vertices that are not in the components containing the hitting set verstices
    disconnected_vertices = np.where(np.isin(labels, components_of_omega, invert=True))[0]
    omega_aug = np.array([o-len(np.where(disconnected_vertices < o)[0]) for o in omega])

    # find the union of the hitting set vertices and disconnected component vertices
    v_comp = np.setdiff1d(np.array(range(A.shape[0])), disconnected_vertices)

    # create augmented adjacency matrix, removing vertices
    A_aug = A[v_comp, :][:, v_comp]

    return A_aug, omega_aug, disconnected_vertices


def hitting_time_moments(A, omega, tol=1e-8, maxiter=5, x0=None):
    """
    Computation of hitting time moments using a linear algebraic approach,
    as defined in the following paper:

    A Deterministic Hitting-Time Moment Approach to Seed-set Expansion over a Graph,
    A. Foss, R.B. Lehoucq, Z.W. Stuart, J.D.Tucker, J.W. Berry,
    https://arxiv.org/pdf/2011.09544.pdf

    Parameters
    ----------
        A: :class:`scipy.sparse.coo_matrix`
            Sparse adjacency matrix, size n x n.
        omega: :class:`numpy.ndarray`
            Indices of hitting set vertices, size o.
        tol: :class:`float`
            Stopping tolerance for conjugate gradient solver (Default: 1e-8)
        maxiter: :class:`int`
            Maximum number of conjugate gradient iterations to run (Default: 5)
        x0: :class:`numpy.ndarray` or `None`
            Starting vector of conjugate gradient solver (Default: None)

    Returns
    -------
        ETm: :class:`numpy.ndarray`
            Hitting time moments, size n x 2. All moments
            associated with indices in `omega` will be equal to 0.

    Example
    -------
    Compute the first two hitting time moments:

    >>> # adjacency matrix
    >>> n = 5
    >>> row = np.array([0, 1, 1, 2, 3])
    >>> col = np.array([1, 2, 3, 4, 4])
    >>> data = np.array([1, 1, 1, 1, 1])
    >>> A = coo_matrix((data, (row, col)), shape=(n, n))
    >>> A = A + A.T
    >>> # hitting set indices
    >>> omega = [0]
    >>> # compute hitting time moments
    >>> ETm, cg_info = hitmix.hitting_time_moments_rl(A, omega)
    >>> print(ETm)
    [[ 9.     10.583 ]
    [12.     10.9545]
    [12.     10.9545]
    [13.     10.9545]]
    """

    # input matrix size
    m, n = A.shape
    assert m == n, "Input matrix 'A' must be square"

    # number of moments to compute
    num_moments = 2

    # extract augmented matrix
    A_aug, omega_aug, disconnected_vertices = augmented_adjacency(A, omega)
    m_aug, n_aug = A_aug.shape

    # diagonal degree matrix
    D_aug = spdiags(A_aug.sum(axis=0), 0, n_aug, n_aug)

    # storage for hitting time moments
    ETm = np.zeros((n_aug, num_moments))

    # information return from CG solver
    cg_info = np.zeros((num_moments))

    # indices of hitting set complement
    omega_c_aug = np.setdiff1d(np.array(range(n_aug)), np.array(omega_aug))

    # first (raw) moment, equivalent to Equation (5) using CG solve and preconditioner D
    rhs_0 = np.zeros((n_aug, 1))
    rhs_0[omega_c_aug] = (D_aug @ np.ones((n_aug, 1)))[omega_c_aug]
    ETm[:, 0], cg_info[0] = cg(prolongation_restriction_matvec(D_aug, A_aug, omega_c_aug), rhs_0, M=D_aug, x0=x0, tol=tol, maxiter=maxiter, atol='legacy')

    # second (raw) moment, equivalent to Equation (5) using CG solve and preconditioner D
    rhs_1 = np.zeros((n_aug, 1))
    rhs_1[omega_c_aug, 0] = (D_aug @ ETm[:, 0] + A_aug @ ETm[:, 0])[omega_c_aug]
    ETm[:, 1], cg_info[1] = cg(prolongation_restriction_matvec(D_aug, A_aug, omega_c_aug), rhs_1, M=D_aug, x0=x0, tol=tol, maxiter=maxiter, atol='legacy')

    # Center the second raw moments about the mean hitting times
    # Note: \mu_i = ET^1_i and \sigma_i = sqrt(E(T^2i - \m_i))
    ETm[:, 1] = ETm[:, 1] - np.power(ETm[:, 0], 2)
    ETm[:, 1] = np.sqrt(ETm[:, 1])

    # storage for full hitting time moments
    full_ETm = np.zeros((n, num_moments))
    aug_verts = np.setdiff1d(np.array(range(n)), disconnected_vertices)
    full_ETm[aug_verts, :] = ETm[:, :]
    full_ETm[disconnected_vertices, :] = math.inf

    return full_ETm, cg_info
