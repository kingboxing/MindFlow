from __future__ import print_function
from fenics import *
import numpy as np
import numbers
import scipy.sparse.linalg as spla
from scipy.sparse import issparse,lil_matrix, dok_matrix,SparseEfficiencyWarning
from scipy import linalg
"""
Created by Bo Jin, 2019.
"""    
    
"""
This module provides classes that define randomized SVD for sparse matrices/linearoperator
 
a modified version of RSVD in scikit-learn
"""
def isoperator(x):
    """Is x of a linear operator?
    Parameters
    ----------
    x
        object to check for being a linear operator
    Returns
    -------
    bool
        True if x is a linear operator, False otherwise
    Notes
    -----
    """
    return isinstance(x, spla.LinearOperator)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
                     
def safe_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix/linear operator case correctly
    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.
    Parameters
    ----------
    a : array or sparse matrix or linear operator 
    b : array or sparse matrix or linear operator 
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.
    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output=False``.
    """
    if issparse(a):
        if isoperator(b): 
            ret = spla.aslinearoperator(a) * b
            if dense_output:
                raise ValueError("'b' should not be a linear operator")
        else:
            ret = a * b
            if dense_output and hasattr(ret, "toarray"):
                ret = ret.toarray()
            return ret
            
    elif issparse(b):
        if isoperator(a):
            ret = a * spla.aslinearoperator(b) 
            if dense_output:
                raise ValueError("'a' should not be a linear operator")
            return ret
        else:
            ret = a * b
            if dense_output and hasattr(ret, "toarray"):
                ret = ret.toarray()
            return ret
    elif isoperator(a):
        if isoperator(b):
            ret = a * b
            if dense_output:
                raise ValueError("'a' and 'b' should not be a linear operator")
            return ret
        else:
            ret = a * b
            #ret = a.matmat(b)
            return ret
    elif isoperator(b):
        ret = np.transpose(b.T * np.transpose(a))
#        a_H=np.transpose(np.conj(a))
#        ret = np.hstack([b.rmatvec(col.reshape(-1,1)) for col in a_H.T])
#        ret = np.transpose(np.conj(ret))
        return ret
    else:
        return np.dot(a, b)

def randomized_range_finder(A, size, n_iter,power_iteration_normalizer='auto',random_state=None):
    
    """Computes an orthonormal matrix whose range approximates the range of A.
    Parameters
    ----------
    A : 2D array/ linear operator 
        The input data matrix
    size : integer
        Size of the return array
    n_iter : integer
        Number of power iterations used to stabilize the result
    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.
        .. versionadded:: 0.18
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    Returns
    -------
    Q : 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.
    Notes
    -----
    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf
    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))
    if A.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)
    elif A.dtype.kind == 'c':
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False) + random_state.normal(size=(A.shape[1], size))*1j

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if power_iteration_normalizer == 'none':
            Q = safe_dot(A, Q)
            Q = safe_dot(A.H, Q)
        elif power_iteration_normalizer == 'LU':
            Q, _ = linalg.lu(safe_dot(A, Q), permute_l=True)
            Q, _ = linalg.lu(safe_dot(A.H, Q), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            Q, _ = linalg.qr(safe_dot(A, Q), mode='economic')
            Q, _ = linalg.qr(safe_dot(A.H, Q), mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(safe_dot(A, Q), mode='economic')
    return Q
    
def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.
    v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v
   
    
    
def randomized_svd(M, n_components, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=0):
    """Computes a truncated randomized SVD
    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose
    n_components : int
        Number of singular values and vectors to extract.
    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
        .. versionchanged:: 0.18
    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.
        .. versionadded:: 0.18
    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
        .. versionchanged:: 0.18
    flip_sign : boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).
    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061
    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    if isinstance(M, (lil_matrix, dok_matrix)):
        warnings.warn("Calculating SVD of a {} is expensive. "
                      "csr_matrix is more efficient.".format(
                          type(M).__name__),SparseEfficiencyWarning)

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples 
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(M, n_random, n_iter,
                                power_iteration_normalizer, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_dot(Q.T.conj(), M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)

    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, V = svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].H, s[:n_components], U[:, :n_components].H
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]