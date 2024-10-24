#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a collection of utility functions for linear algebra operations, sparse matrix manipulations,
eigenvalue computations, and working with FEniCS functions. These utilities are intended to assist in finite element
simulations and numerical methods involving linear algebra.

Functions
---------
- assign2(receiving_func, assigning_func):
    Assign a NumPy array or another FEniCS function to a target FEniCS function.
- allclose_spmat(A, B, tol=1e-12):
    Check if two sparse matrices A and B are identical within a tolerance.
- get_subspace_info(function_space):
    Get the number of scalar subspaces under each top-level subspace in a given function space.
- find_subspace_index(index, sub_spaces):
    Find the top-level and sub-level subspace indices for a given scalar subspace index.
- is_symmetric(m):
    Check if a sparse matrix is symmetric.
- del_zero_cols(mat):
    Delete columns in a sparse matrix that have all elements equal to zero.
- eigen_decompose(A, M=None, k=3, sigma=0.0, solver_params=None):
    Perform eigenvalue decomposition on the system λ*M*x = A*x or A*x = λ*x.
- sort_complex(a, tol=1e-8):
    Sort a complex array based on the real part, then the imaginary part.
- distribute_numbers(n, k):
    Distribute n numbers into k groups as evenly as possible.
- convert_to_2d(arr, axis=0):
    Convert a 1D array to a 2D array.
- save_complex(complex_list, filename):
    Save a list of complex numbers to a text file.
- load_complex(filename):
    Load a list of complex numbers from a text file.
- plot_spmat(sparse_matrix):
    Plot the non-zero elements of a sparse matrix.
- rmse(predictions, targets):
    Compute the root-mean-square error between two arrays.
- dict_deep_update(original, updates):
    Recursively update a dictionary with another dictionary.
- deep_set_attr(obj, attr):
    Recursively set attributes on an object based on a nested dictionary.
- assemble_sparse(blocks):
    Assemble a sparse matrix from a list of block matrices.
- assemble_dae2(model):
    Assemble state-space matrices from block matrices for a DAE2 system.
- is_diag_sparse(sparse_matrix):
    Check if a sparse matrix is diagonal.
- find_block_boundaries(sparse_matrix):
    Find block boundaries in a sparse block diagonal matrix.
- extract_diagonal_blocks(sparse_matrix):
    Extract diagonal blocks from a sparse block diagonal matrix.
- invert_dense_blocks(blocks):
    Compute the inverse of each dense block matrix in a list.
- assemble_diag_block_matrix(inverted_blocks):
    Assemble inverted blocks into a sparse block diagonal matrix.
- invert_diag_block_matrix(sparse_matrix, maxsize=3000):
    Invert a sparse block diagonal matrix.
- cholesky_sparse(sparse_matrix, maxsize=3000):
    Compute the Cholesky decomposition of a sparse symmetric positive-definite matrix.
- woodbury_solver(U, V, b):
    Solve a linear system using the Woodbury matrix identity.
- find_orthogonal_complement(A, U, M=None, tolerance=1e-6):
    Find an orthogonal basis that complements U and improves the representation of A.

Notes
-----
- Ensure that the required dependencies are installed, such as NumPy, SciPy, matplotlib, and FEniCS.
- Some functions require specific formats for matrices (e.g., CSR or CSC sparse matrices).
"""
from ..Deps import *


def assign2(receiving_func, assigning_func):
    """
    Assign a NumPy array or another FEniCS function to a target FEniCS function.

    Parameters
    ----------
    receiving_func : function.function.Function
        The target FEniCS function to which the values will be assigned.
    assigning_func : np.ndarray, function.function.Function
        The source of the values to be assigned. This can be either:
          - A NumPy array with a size matching the number of DOFs in the target FEniCS function.
          - Another FEniCS function defined on the same function space.

    Raises
    ------
    ValueError
        If the source type is not supported or the sizes do not match.

    Notes
    -----
    This function facilitates the assignment of data to a FEniCS function, ensuring that the data aligns with the function space.

    Examples
    --------
        assign2(u, np_array)
        assign2(u, v)  # where u and v are FEniCS functions

    """
    if isinstance(assigning_func, np.ndarray):
        # Check if the size of the NumPy array matches the number of DOFs in the FEniCS function
        if receiving_func.vector().size() != assigning_func.size:
            raise ValueError(f"Size mismatch: FEniCS function has {receiving_func.vector().size()} DOFs, "
                             f"but the NumPy array has {assigning_func.size} elements.")
        # Assign the NumPy array values to the FEniCS function
        receiving_func.vector()[:] = np.ascontiguousarray(assigning_func)
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

    Parameters
    ----------
    A : scipy.sparse matrix
        The first sparse matrix.
    B : scipy.sparse matrix
        The second sparse matrix.
    tol : float, optional
        Tolerance value for the comparison. Default is 1e-12.

    Returns
    -------
    bool
        True if matrices are identical within the specified tolerance, otherwise False.

    Notes
    -----
    This function computes the Frobenius norm of the difference between A and B and compares it with the tolerance.

    Examples
    --------
         allclose_spmat(A, B, tol=1e-8)
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
    Get the number of scalar subspaces under each top-level subspace in a given function space.

    Parameters
    ----------
    function_space : FunctionSpace
        The function space to analyze.

    Returns
    -------
    sub_spaces : tuple
        A tuple where each entry corresponds to the number of scalar subspaces in a top-level subspace.

    Notes
    -----
    This function is useful for understanding the structure of mixed or vector function spaces.

    Examples
    --------
         sub_spaces = get_subspace_info(mixed_space)
    """
    sub_spaces = ()
    num_sub_spaces = function_space.num_sub_spaces()
    for i in range(num_sub_spaces):
        num_space = function_space.sub(i).num_sub_spaces()
        if num_space == 0:
            sub_spaces += (1,)
            #total_scalar_subspaces += 1
        else:
            sub_spaces += (num_space,)
            #total_scalar_subspaces += num_space

    return sub_spaces


def find_subspace_index(index, sub_spaces):
    """
    Find the top-level subspace and sub-level subspace indices for a given scalar subspace index.

    Parameters
    ----------
    index : int
        The index of the scalar subspace (0-based index).
    sub_spaces : tuple
        A tuple where each entry corresponds to the number of scalar subspaces in a top-level subspace.
        Obtained from 'get_subspace_info(function_space)'.

    Returns
    -------
    indices : tuple
        A tuple with the top-level subspace index and the sub-level subspace index (both 0-based).
        If the top-level subspace contains only one scalar subspace, only the top-level index is returned.

    Raises
    ------
    ValueError
        If the index is out of bounds.

    Examples
    --------
        indices = find_subspace_index(3, sub_spaces)
    """
    cumulative_sum = 0

    for i, num_scalar_subspaces in enumerate(sub_spaces):
        previous_sum = cumulative_sum
        cumulative_sum += num_scalar_subspaces
        if index < cumulative_sum:
            top_level_index = i
            sub_level_index = index - previous_sum
            if sub_spaces[top_level_index] > 1:
                return (top_level_index, sub_level_index)
            else:  # the number of scalar subspaces in a top-level subspace = 1
                return (top_level_index,)

    raise ValueError("Index out of bounds")


def is_symmetric(m):
    """
    Check if a sparse matrix is symmetric.

    Parameters
    ----------
    m : scipy.sparse matrix
        A square sparse matrix.

    Returns
    -------
    bool
        True if the matrix is symmetric, False otherwise.

    Raises
    ------
    ValueError
        If the input matrix is not square.

    Examples
    --------
        symmetric = is_symmetric(sparse_matrix)
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
    Delete columns in a sparse matrix that have all elements equal to zero.

    Parameters
    ----------
    mat : scipy.sparse matrix
        The sparse matrix (CSR or CSC format) from which zero columns will be removed.

    Returns
    -------
    scipy.sparse matrix
        The sparse matrix with zero columns removed.

    Examples
    --------
        mat_nonzero = del_zero_cols(mat)
    """
    return mat[:, mat.nonzero()[1]]


def eigen_decompose(A, M=None, k=3, sigma=0.0, solver_params=None, Mat=None):
    """
    Perform eigenvalue decomposition on the system λ*M*x = A*x or A*x = λ*x.

    Parameters
    ----------
    A : scipy.sparse matrix
        The system matrix.
    M : scipy.sparse matrix, optional
        The mass matrix. If None, the eigenvalue problem A*x = λ*x is solved.
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
    Mat : scipy.sparse matrix or dict with keys 'U' and 'V', optional, pending
        Feedback matrix `Mat = U * V.T`. Can be provided as a sparse matrix or a dictionary containing 'U' and 'V'. Default is None.

    Returns
    -------
    vals : numpy.ndarray
        Array of computed eigenvalues.
    vecs : numpy.ndarray
        Array of computed eigenvectors.

    Notes
    -----
    This function leverages scipy's sparse eigenvalue solvers and supports shift-invert mode.

    Examples
    --------
        vals, vecs = eigen_decompose(A, M, k=5, sigma=0.1)
    """
    # import functions
    from ..Params.Params import DefaultParameters
    from ..LinAlg.MatrixOps import InverseMatrixOperator
    # Default parameters for eigenvalue solver
    default_params = DefaultParameters().parameters['eigen_decompose']
    if solver_params is None:
        solver_params = {}

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


def sort_complex(a, tol=1e-8):
    """
    Sort a complex array based on the real part, then the imaginary part.

    Parameters
    ----------
    a : array_like
        Input complex array to be sorted.
    tol : float, optional
        Precision of array's real part to sort. Default is 1e-8.

    Returns
    -------
    sorted_array : numpy.ndarray
        The input array sorted in descending order, first by real part, then by imaginary part.
    index_sort : numpy.ndarray
        Indices that sort the original array in descending order.

    Examples
    --------
        sorted_array, index_sort = sort_complex(complex_array)
    """
    # Get the indices that would sort the array based on real part and then imaginary part
    atol = np.round(a.real, int(np.abs(np.log10(tol)))) + 1j * a.imag
    index_sort = np.lexsort((atol.imag, atol.real))[::-1]  # Reverse for descending order

    # Use the sorted indices to get the sorted array
    sorted_array = a[index_sort]

    return sorted_array, index_sort


def distribute_numbers(n, k):
    """
    Distribute n numbers into k groups as evenly as possible.

    Parameters
    ----------
    n : int
        The total number of elements to distribute.
    k : int
        The number of groups.

    Returns
    -------
    list of int
        A list where each element represents the number of elements in that group.

    Examples
    --------
        distribution = distribute_numbers(10, 3)
    """
    # Base size of each group
    base_size = n // k

    # Number of groups that will get an extra element
    remainder = n % k

    # Create the distribution list
    distribution = [base_size + 1 if i < remainder else base_size for i in range(k)]

    return distribution


def convert_to_2d(arr, axis=0):
    """
    Convert a 1D array to a 2D array.

    Parameters
    ----------
    arr : array_like
        Input array to check and potentially convert.
    axis : int, optional
        The axis along which to expand the dimensions. Default is 0.

    Returns
    -------
    numpy.ndarray
        A 2D version of the input array.

    Examples
    --------
        arr_2d = convert_to_2d(arr, axis=1)

    Check if an array is 1-dimensional, and if yes, convert it to 2D.
    """
    arr = np.asarray(arr)  # Ensure input is a NumPy array

    if arr.ndim == 1:  # Check if the array is 1D
        arr = np.expand_dims(arr, axis=axis)  # Convert to 2D (row vector)

    return arr


def save_complex(complex_list, filename):
    """
    Save a list of complex numbers to a text file.

    Parameters
    ----------
    complex_list : list of complex
        List of complex numbers to store.
    filename : str
        Name of the file to write to.

    Examples
    --------
        save_complex(eigenvalues, 'eigenvalues.txt')
    """
    with open(filename, 'w') as file:
        for num in complex_list:
            # Write real and imaginary parts separately
            file.write(f"{num.real} {num.imag}\n")


def load_complex(filename):
    """
    Load a list of complex numbers from a text file.

    Parameters
    ----------
    filename : str
        Name of the file to read from.

    Returns
    -------
    list of complex
        List of complex numbers.

    Examples
    --------
        complex_list = load_complex('eigenvalues.txt')
    """
    complex_list = []
    with open(filename, 'r') as file:
        for line in file:
            real, imag = map(float, line.split())
            complex_list.append(complex(real, imag))
    return complex_list


def plot_spmat(sparse_matrix):
    """
    Plot the non-zero elements of a scipy sparse matrix.

    Parameters
    ----------
    sparse_matrix : scipy.sparse matrix
        The sparse matrix to plot.

    Examples
    --------
        plot_spmat(A)
    """
    # Convert the sparse matrix to COO format for easy access to row, col, and data
    sparse_coo = sparse_matrix.tocoo()

    # Plot the non-zero elements as a scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(sparse_coo.col, sparse_coo.row, marker='o', color='blue', s=5)  # Use row, col coordinates
    plt.title('Non-zero Elements of the Sparse Matrix')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.gca().invert_yaxis()  # Invert y-axis to match matrix layout
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
    plt.show()


def rmse(predictions, targets):
    """
    Compute the root-mean-square error between two arrays.

    Parameters
    ----------
    predictions : array_like
        Predicted values.
    targets : array_like
        True values.

    Returns
    -------
    float
        The root-mean-square error.

    Examples
    --------
        error = rmse(predicted_values, true_values)
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def dict_deep_update(original, updates):
    """
    Recursively update the original dictionary with the updates dictionary.

    Parameters
    ----------
    original : dict
        The original dictionary to be updated.
    updates : dict
        The updates to apply.

    Returns
    -------
    dict
        The updated dictionary.

    Examples
    --------
        updated_dict = dict_deep_update(original_dict, updates_dict)
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            # If both the original and update value are dicts, recurse into the dicts
            dict_deep_update(original[key], value)
        else:
            # Otherwise, simply update/overwrite the value
            original[key] = value
    return original


def deep_set_attr(obj, attr):
    """
    Recursively set attributes on an object based on a nested dictionary.

    Parameters
    ----------
    obj : object
        The object on which attributes will be set.
    attr : dict
        A dictionary representing the attributes and their values. Nested dictionaries represent nested attributes.

    Examples
    --------
        deep_set_attr(my_object, {'a': 1, 'b': {'c': 2}})
    """
    for key, value in attr.items():
        if isinstance(value, dict):
            # Recursively set attributes on nested objects
            nested_obj = getattr(obj, key, None)
            if nested_obj is None:
                nested_obj = type('DynamicObject', (object,), {})()  # Create a new dynamic object if None
                setattr(obj, key, nested_obj)
            deep_set_attr(nested_obj, value)
        else:
            # Set the attribute directly if the value is not a dictionary
            setattr(obj, key, value)


def assemble_sparse(blocks):
    """
    Assemble a sparse matrix from a list of block matrices.

    Parameters
    ----------
    blocks : list of lists
        A 2D list of block matrices (either sparse or dense). Each row in the list represents a row of block matrices.
        Use None for empty blocks.

    Returns
    -------
    scipy.sparse matrix
        A single assembled sparse matrix.

    Examples
    --------
        A_full = assemble_sparse([[A, B], [C, None]])
    """
    # Convert any dense NumPy arrays in the blocks to sparse format
    sparse_blocks = [[sp.csr_matrix(block) if block is not None and not sp.issparse(block) else block
                      for block in row]
                     for row in blocks]
    # Use bmat to assemble the sparse matrix from blocks
    A = sp.bmat(sparse_blocks).tocsr()
    A.eliminate_zeros()
    A.sort_indices()  # sort data indices, prevent unmfpack error -8
    return A


def assemble_dae2(model):
    """
    Assemble state-space matrices from block matrices for a DAE2 system.

    Parameters
    ----------
    model : dict
        A dictionary containing the following keys:
            - 'A': The system matrix.
            - 'G': The coupling matrix.
            - 'M': The mass matrix.
            Mass = E_full = | M   0 |      State = A_full = | A   G  |
                            | 0   0 |                       | G.T Z=0|

    Returns
    -------
    A_full : scipy.sparse matrix
        The assembled state matrix.
    E_full : scipy.sparse matrix
        The assembled mass matrix.

    Examples
    --------
        A_full, E_full = assemble_dae2(model)
    """
    # Get the dimensions of G
    n = model['G'].shape[1]

    # Assemble the state matrix A_full using the blocks [A, G] and [G.T, None]
    A_full = assemble_sparse([
        [model['A'], model['G']],
        [model['G'].T, None]
    ])

    # Assemble the mass matrix E_full using the blocks [M, None] and [None, zeros(n, n)]
    E_full = assemble_sparse([
        [model['M'], None],
        [None, sp.csr_matrix((n, n))]  # Zeros block
    ])

    return A_full, E_full


def is_diag_sparse(sparse_matrix):
    """
    Check if a sparse matrix is a diagonal matrix.

    Parameters
    ----------
    sparse_matrix : scipy.sparse matrix
        The sparse matrix to check.

    Returns
    -------
    bool
        True if the matrix is diagonal, False otherwise.

    Examples
    --------
        is_diagonal = is_diag_sparse(sparse_matrix)
    """
    # Convert the matrix to COO format (for efficient access to non-zero elements)
    coo_matrix = sparse_matrix.tocoo()

    # Check if all non-zero elements are on the diagonal
    return np.all(coo_matrix.row == coo_matrix.col)


def find_block_boundaries(sparse_matrix):
    """
    Find the block boundaries in a sparse block diagonal matrix based on the index pattern of non-zero elements.

    Parameters
    ----------
    sparse_matrix : scipy.sparse matrix
        The sparse matrix (CSR format) to analyze.

    Returns
    -------
    block_boundaries : list of int
        A list of indices indicating where the blocks start and end.

    Examples
    --------
        boundaries = find_block_boundaries(sparse_matrix)
    """
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    n = sparse_matrix.shape[0]

    # Initialize block boundaries list
    block_boundaries = [0]

    # Find where each block ends by checking the row-wise non-zero pattern
    for i in range(1, n):
        # Check if row i has any non-zero elements before column i
        row_nonzeros = sparse_matrix.indices[sparse_matrix.indptr[i]:sparse_matrix.indptr[i + 1]]

        # If there are no non-zeros in the current row before column i, it indicates a new block
        if len(row_nonzeros) == 0 or np.all(row_nonzeros >= i):
            block_boundaries.append(i)

    block_boundaries.append(n)
    return block_boundaries


def extract_diagonal_blocks(sparse_matrix):
    """
    Extract diagonal blocks from a sparse block diagonal matrix based on its non-zero pattern.

    Parameters
    ----------
    sparse_matrix : scipy.sparse matrix
        The sparse matrix (CSR format) with a block diagonal structure.

    Returns
    -------
    blocks : list of numpy.ndarray
        A list of dense matrices representing the diagonal blocks.

    Examples
    --------
        blocks = extract_diagonal_blocks(sparse_matrix)
    """
    # Ensure the matrix is in CSR format for efficient row-wise operations
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    # Find block boundaries
    block_boundaries = find_block_boundaries(sparse_matrix)

    # Extract the diagonal blocks
    blocks = []
    for i in range(len(block_boundaries) - 1):
        start = block_boundaries[i]
        end = block_boundaries[i + 1]

        # Extract the submatrix (block) and convert it to a dense format
        block = sparse_matrix[start:end, start:end].toarray()
        blocks.append(block)

    return blocks


def invert_dense_blocks(blocks):
    """
    Compute the inverse of each dense block matrix in the list.

    Parameters
    ----------
    blocks : list of numpy.ndarray
        A list of dense matrices.

    Returns
    -------
    inverted_blocks : list of numpy.ndarray
        A list of inverted dense matrices.

    Examples
    --------
        inverted_blocks = invert_dense_blocks(blocks)
    """
    inverted_blocks = []
    for block in blocks:
        # Invert the block using NumPy's inverse function
        block_inv = np.linalg.inv(block)
        inverted_blocks.append(block_inv)
    return inverted_blocks


def assemble_diag_block_matrix(inverted_blocks):
    """
    Assemble the inverted blocks into a sparse block diagonal matrix.

    Parameters
    ----------
    inverted_blocks : list of numpy.ndarray
        A list of inverted dense matrices.

    Returns
    -------
    scipy.sparse matrix
        A sparse matrix with the inverted blocks on its diagonal.

    Examples
    --------
        inv_sparse_matrix = assemble_diag_block_matrix(inverted_blocks)
    """
    # Use scipy.sparse.block_diag to create a block diagonal sparse matrix
    sparse_block_diag = sp.block_diag(inverted_blocks)
    return sparse_block_diag


def invert_diag_block_matrix(sparse_matrix, maxsize=3000):
    """
    Invert a sparse block diagonal matrix.

    The function extract diagonal blocks from a sparse block diagonal matrix, compute the inverse of each block,
    and assemble the inverted blocks into a sparse block diagonal matrix.

    Parameters
    ----------
    sparse_matrix : scipy.sparse matrix
        The sparse matrix (CSR format) with a block diagonal structure.
    maxsize : int, optional
        The maximum size for dense inversion. Default is 3000.

    Returns
    -------
    scipy.sparse matrix or numpy.ndarray
        The inverted matrix.

    Examples
    --------
        inv_matrix = invert_diag_block_matrix(sparse_matrix)
    """
    if not sp.issparse(sparse_matrix):
        sparse_matrix = sp.csr_matrix(sparse_matrix)
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    if sparse_matrix.shape[0] < maxsize:
        sparse_inverted_block_matrix = np.linalg.inv(sparse_matrix.toarray())
    else:
        # Step 1: Extract diagonal blocks
        blocks = extract_diagonal_blocks(sparse_matrix)
        # Step 2: Invert the dense blocks
        inverted_blocks = invert_dense_blocks(blocks)
        # Step 3: Assemble the inverted blocks into a sparse block diagonal matrix
        sparse_inverted_block_matrix = assemble_diag_block_matrix(inverted_blocks)

    return sparse_inverted_block_matrix


def cholesky_sparse(sparse_matrix, maxsize=3000):
    """
    Compute the Cholesky decomposition A = L * L' of a sparse symmetric positive-definite matrix.

    Parameters
    ----------
    sparse_matrix : scipy.sparse matrix
        The sparse symmetric positive-definite matrix.
    maxsize : int, optional
        The maximum size for dense decomposition. Default is 3000.

    Returns
    -------
    scipy.sparse matrix or numpy.ndarray
        The Cholesky factor of the input matrix.

    Raises
    ------
    ValueError
        If the matrix is not symmetric positive-definite.

    Examples
    --------
        L = cholesky_sparse(sparse_matrix)
    """
    # Ensure the matrix is sparse and in CSC format
    if not sp.issparse(sparse_matrix):
        sparse_matrix = sp.csc_matrix(sparse_matrix)

    if not sp.isspmatrix_csc(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsc()

    if is_diag_sparse(sparse_matrix):
        # Get the diagonal elements (non-zero values in a diagonal matrix)
        diagonal = sparse_matrix.diagonal()
        # Invert the diagonal elements (take the reciprocal of each non-zero element)
        diagonal_chol = np.sqrt(diagonal)
        # Update the matrix with the inverted diagonal elements
        sparse_matrix_chol = sp.diags(diagonal_chol, format='csr')
    elif is_symmetric(sparse_matrix):
        if sparse_matrix.shape[0] < maxsize:
            sparse_matrix_chol = np.linalg.cholesky(sparse_matrix.roarray())
        else:
            sparse_matrix_chol = cholesky(sparse_matrix, ordering_method="natural").L()
    else:
        raise ValueError("Sparse matrix is not a diagonal or a symmetric positive definite matrix.")

    return sparse_matrix_chol


def woodbury_solver(U, V, b):
    """
    Solve the system (I + U * V^T) x = b using the Woodbury matrix identity.

    The Woodbury formula is:
    (I + U * V^T)^-1 = I - U * (I + V^T * U)^-1 * V^T

    Parameters
    ----------
    U : numpy.ndarray
        Matrix U in the Woodbury identity (shape (n, k)).
    V : numpy.ndarray
        Matrix V in the Woodbury identity (shape (n, k)).
    b : numpy.ndarray
        Right-hand side vector or matrix for the system (shape (n,) or (n, 1)).

    Returns
    -------
    x : numpy.ndarray
        The solution to the system (I + U * V^T) x = b.

    Raises
    ------
    ValueError
        If the shapes of U, V, and b are incompatible.

    Examples
    --------
        x = woodbury_solver(U, V, b)
    """
    n, k = U.shape

    # Ensure V has the right shape
    if V.shape == (k, n):
        V = V.T
    elif V.shape != (n, k):
        raise ValueError("V.T and U must have same shape")
    # Ensure b is a column vector
    if b.shape == (n,):
        b = b.reshape(n, 1)
    elif b.shape == (1, n):
        b = b.T
    elif b.shape != (n, 1):
        raise ValueError("b must have the shape (n,) or (n, 1)")

    x = b - U @ (np.linalg.inv(np.identity(k) + V.T @ U) @ V.T @ b)
    return x.flatten()


import numpy as np


def find_orthogonal_complement(A, U, M=None, tolerance=1e-6):
    """
    Find an orthogonal basis that complements U and improves the representation of A.
    A truncation tolerance is given to filter out insignificant modes.
    Note that U can be orthogonal with respect to the weight matrix M.

    Similar to Gram-Schmidt orthogonalization

    Parameters
    ----------
    A : numpy.ndarray
        The matrix containing k modes (shape (n, k)).
    U : numpy.ndarray
        The initial orthogonal basis of l modes (shape (n, l)).
    M : scipy.sparse matrix, optional
        The sparse weight matrix. If None, the identity matrix is used.
    tolerance : float, optional
        Relative truncation tolerance for filtering insignificant modes based on singular values.

    Returns
    -------
    new_basis : numpy.ndarray of shape (n, l + r_filtered)
        The combined orthogonal basis, consisting of the original U and new basis vectors.

    Examples
    --------
        new_basis = find_orthogonal_complement(A, U, tolerance=1e-5)
    """
    if M is None:
        M = sp.identity(A.shape[0])
    # Step 1: Project A onto U
    U_proj = np.linalg.inv(U.T @ M @ U) @ U.T  # normalised mode
    A_proj = U @ (U_proj @ M @ A)  # represent A by U

    # Step 2: Find the residual
    R = A - A_proj

    # Step 3: Perform SVD on the residual to find the orthogonal complement
    # Form the eigenvalue problem (equivalent to SVD)
    R_cov = R.T @ M @ R
    eigvals, eigvecs = np.linalg.eigh(R_cov)  # Solve the eigenvalue problem
    sig = np.diag(np.reciprocal(np.sqrt(eigvals)))

    # Step 4: Apply the traction tolerance to filter out small singular values
    significant_modes = np.sqrt(eigvals) > tolerance * np.sqrt(np.sum(eigvals))
    U_hat_filtered = (R @ eigvecs @ sig)[:, significant_modes] # Keep only significant modes

    # Step 5: Combine the original U with the filtered new orthogonal complement
    new_basis = np.hstack((U, U_hat_filtered))

    return new_basis
