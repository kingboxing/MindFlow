#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:48:11 2024

@author: bojin
"""
from src.Deps import *
from src.LinAlg.MatrixOps import InverseMatrixOperator

def assign2(receiving_func, assigning_func):
    """
    Assigns a NumPy array or another FEniCS function to a target FEniCS function.

    Parameters
    ----------
    receiving_func : function.function.Function
        The recieving function. The target FEniCS function to which the values will be assigned.
    assigning_func : np.ndarray, function.function.Function
        The assigning function. The source of the values to be assigned. This can be either:
          - A NumPy array with a size matching the number of DOFs in the target FEniCS function.
          - Another FEniCS function defined on the same function space.

    Raises:
    - ValueError: If the source type is not supported or the sizes do not match.

    """
    if isinstance(assigning_func, np.ndarray):
        # Check if the size of the NumPy array matches the number of DOFs in the FEniCS function
        if receiving_func.vector().size() != assigning_func.size:
            raise ValueError(f"Size mismatch: FEniCS function has {fenics_function.vector().size()} DOFs, "
                             f"but the NumPy array has {source.size} elements.")
        # Assign the NumPy array values to the FEniCS function
        receiving_func.vector()[:]=np.ascontiguousarray(assigning_func)
    elif isinstance(assigning_func, function.function.Function):
        # Check if the function spaces match
        if assigning_func.function_space() != receiving_func.function_space():
            raise ValueError("Function spaces do not match.")
        receiving_func.assign(assigning_func)
    else:
        raise ValueError("Unsupported source type. Must be a NumPy array or a FEniCS function.")

def allclose_spmat(A, B, tol=1e-12):
    """
    Check if two sparse matrices A and B are identical within a tolerance.

    Parameters:
    - A, B: scipy.sparse matrices
    - tol: tolerance value

    Returns:
    - True if matrices are identical within the specified tolerance, otherwise False.
    """
    if A.shape != B.shape:
        return False

    # Compute the difference between the matrices
    diff = A - B

    # Compute the norm of the difference
    diff_norm = spla.norm(diff, ord='fro')

    # Check if the norm is within the tolerance
    return diff_norm <= tol


def get_subspace_info(function_space):
    """
    Get the number of top-level sub-elements and the total number of scalar subspaces 
    in a given two-level function space.

    Parameters
    ----------
    function_space : FunctionSpace
        DESCRIPTION.

    Returns
    -------
    sub_spaces : tuple
        The number of scalar subspaces under each subspace.

    """
    sub_spaces = ()
    num_sub_spaces = function_space.num_sub_spaces()
    for i in range(num_sub_spaces):
        num_space = function_space.sub(i).num_sub_spaces()
        if num_space == 0:
            sub_spaces += (1, )
            #total_scalar_subspaces += 1
        else:
            sub_spaces += (num_space, )
            #total_scalar_subspaces += num_space
    
    return sub_spaces

def find_subspace_index(index, sub_spaces):
    """
    Find the top-level subspace and sub-level subspace indices for a given scalar subspace index.

    Parameters
    ----------
    index : int
        The index of the scalar subspace (0-based index)..
    sub_spaces : tuple
        number scalar subspaces per top level subspace. 
        A list where each entry corresponds to the number of scalar subspaces in a top-level subspace.
        obtained from 'get_subspace_info(function_space)'

    Raises
    ------
    ValueError
        Index out of bounds.

    Returns
    -------
    (top_level_index, sub_level_index) : tuple
        A tuple with the top-level subspace index and the sub-level subspace index (both 0-based)..

    """
    cumulative_sum = 0
    
    for i, num_scalar_subspaces in enumerate(sub_spaces):
        previous_sum = cumulative_sum
        cumulative_sum += num_scalar_subspaces
        if index < cumulative_sum:
            top_level_index = i
            sub_level_index = index - previous_sum
            if sub_spaces[top_level_index]>1:
                return (top_level_index, sub_level_index)
            else: # the number of scalar subspaces in a top-level subspace = 1
                return (top_level_index, )
                
    raise ValueError("Index out of bounds")
    
def is_symmetric(m):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, sp.coo_matrix):
        m = sp.coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


def del_zero_cols(mat):
    """
    Deletes columns in a sparse matrix with all elements equal to zero

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix/csc_matrix
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return mat[:, mat.nonzero()[1]]

def eigen_decompose(A, M=None, k=3, sigma=0.0, solver_params=None):
    """
    Perform eigen-decomposition on the system: λ*M*x = A*x or A*x = λ*x.

    Parameters
    ----------
    A : scipy.sparse matrix
        State matrix.
    M : scipy.sparse matrix, optional
        Mass matrix. If None, A*x = λ*x is solved.
    k : int, optional
        Number of eigenvalues to compute. Default is 3.
    sigma : float, optional
        Shift-invert parameter. Default is 0.0.
    solver_params : dict, optional
        Parameters for the eigenvalue solver, including:
        - method: str (e.g., 'lu')
        - lusolver: str (e.g., 'mumps')
        - echo: bool (default False)
        - which: str (default 'LM')
        - v0: numpy array (default None)
        - ncv: int (default None)
        - maxiter: int (default None)
        - tol: float (default 0)
        - return_eigenvectors: bool (default True)
        - OPpart: None or str (default None)

    Returns
    -------
    vals : numpy array
        Eigenvalues.
    vecs : numpy array
        Eigenvectors.
    """
    if solver_params is None:
        solver_params = {}

    # Default parameters for eigenvalue solver
    default_params = {
        'method': 'lu',
        'lusolver': 'mumps',
        'echo': False,
        'which': 'LM',
        'v0': None,
        'ncv': None,
        'maxiter': None,
        'tol': 0,
        'return_eigenvectors': True,
        'OPpart': None
    }

    # Update default parameters with user-provided ones
    solver_params = {**default_params, **solver_params}

    # Shift-invert operator
    OP = A - sigma * M if sigma else A

    OPinv = None
    if sigma is not None:
        # Shift-invert mode requires an inverse operator
        info('Internal Shift-Invert Mode Solver is active')
        if solver_params['method'] == 'lu':
            info(f"LU decomposition using {solver_params['lusolver'].upper()} solver...")
            OPinv = InverseMatrixOperator(OP, lusolver=solver_params['lusolver'], echo=solver_params['echo'])
            info('Done.')
        else:
            info('Iterative solver is pending development.')

    # Perform eigen-decomposition using scipy.sparse.linalg.eigs
    return spla.eigs(A, k=k, M=M, Minv=None, OPinv=OPinv, sigma=sigma, which=solver_params['which'],
                     v0=solver_params['v0'], ncv=solver_params['ncv'], maxiter=solver_params['maxiter'],
                     tol=solver_params['tol'], return_eigenvectors=solver_params['return_eigenvectors'],
                     OPpart=solver_params['OPpart'])
    
def sort_complex(a):
    """
    Sort a complex array based on the real part, then the imaginary part.
    
    Parameters:
    a : array_like
        Input complex array to be sorted.
        
    Returns:
    sorted_array : ndarray
        The input array sorted in descending order, first by real part, then by imaginary part.
    index_sort : ndarray
        Indices that sort the original array in descending order.
    """
    # Get the indices that would sort the array based on real part and then imaginary part
    index_sort = np.lexsort((a.imag, a.real))[::-1]  # Reverse for descending order
    
    # Use the sorted indices to get the sorted array
    sorted_array = a[index_sort]
    
    return sorted_array, index_sort
    
    
    