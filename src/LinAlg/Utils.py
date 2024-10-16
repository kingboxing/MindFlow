#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:48:11 2024

@author: bojin
"""
from ..Deps import *
from ..LinAlg.MatrixOps import InverseMatrixOperator


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
            if sub_spaces[top_level_index] > 1:
                return (top_level_index, sub_level_index)
            else:  # the number of scalar subspaces in a top-level subspace = 1
                return (top_level_index,)

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
    # Default parameters for eigenvalue solver
    from ..Params.Params import DefaultParameters
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
    
    Parameters:
    a : array_like
        Input complex array to be sorted.
    tol : float, optional
        Precision of array's real part to sort. The default is 1e-8.
        
    Returns:
    sorted_array : ndarray
        The input array sorted in descending order, first by real part, then by imaginary part.
    index_sort : ndarray
        Indices that sort the original array in descending order.
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
    
    Parameters:
    n : int
        The total number of elements to distribute.
    k : int
        The number of groups.
        
    Returns:
    list of int
        A list where each element represents the number of elements in that group.
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
    Check if an array is 1-dimensional, and if yes, convert it to 2D.
    
    Parameters:
    arr : array_like
        Input array to check and potentially convert.
    axis : int
        the axis to expand.
    Returns:
    ndarray
        A 2D version of the input array.
    """
    arr = np.asarray(arr)  # Ensure input is a NumPy array

    if arr.ndim == 1:  # Check if the array is 1D
        arr = np.expand_dims(arr, axis=axis)  # Convert to 2D (row vector)

    return arr


def save_complex(complex_list, filename):
    """
    Save a list of complex numbers to a text file.
    
    Parameters:
    complex_list : list of complex
        List of complex numbers to store.
    filename : str
        Name of the file to write to.
    """
    with open(filename, 'w') as file:
        for num in complex_list:
            # Write real and imaginary parts separately
            file.write(f"{num.real} {num.imag}\n")


def load_complex(filename):
    """
    Load a list of complex numbers from a text file.
    
    Parameters:
    filename : str
        Name of the file to read from.
        
    Returns:
    list of complex
        List of complex numbers.
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
    
    Parameters:
    sparse_matrix : scipy.sparse.csr_matrix
        The sparse matrix to plot.
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
    Root-mean-square deviation between two arrays

    Parameters
    ----------------------------
    predictions : Predicted array

    targets : Obtained array

    Returns
    ----------------------------
    Root-mean-square deviation

    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def dict_deep_update(original, updates):
    """
    Recursively update the original dictionary with the updates dictionary.

    Parameters:
    original (dict): The original dictionary to be updated.
    updates (dict): The updates to apply.

    Returns:
    dict: The updated dictionary.
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

    Parameters:
    - obj: The object on which attributes will be set.
    - attributes: A dictionary representing the attributes and their values. Nested dictionaries represent nested attributes.
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

    Parameters:
    - blocks: A 2D list of block matrices (either sparse or dense).
              Each row in the list represents a row of block matrices.
              Use None for empty blocks.

    Returns:
    - A single assembled sparse matrix.
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

    Mass = E_full = | M   0 |      State = A_full = | A   G  |
                    | 0   0 |                       | G.T Z=0|

    Parameters:
    - model: A dictionary containing the following keys:
      - 'A': The system matrix.
      - 'G': The coupling matrix.
      - 'M': The mass matrix.

    Returns:
    - A tuple (A_full, E_full) where:
      - A_full: The assembled state matrix.
      - E_full: The assembled mass matrix.
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

    Parameters:
    - sparse_matrix: A scipy sparse matrix.

    Returns:
    - True if the matrix is diagonal, False otherwise.
    """
    # Convert the matrix to COO format (for efficient access to non-zero elements)
    coo_matrix = sparse_matrix.tocoo()

    # Check if all non-zero elements are on the diagonal
    return np.all(coo_matrix.row == coo_matrix.col)


def find_block_boundaries(sparse_matrix):
    """
    Find the block boundaries in a sparse block diagonal matrix based on the index pattern of non-zero elements.

    Parameters:
    - sparse_matrix: A sparse matrix in CSR format.

    Returns:
    - block_boundaries: A list of indices indicating where the blocks start and end.
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

    Parameters:
    - sparse_matrix: A sparse matrix (CSR format) with a block diagonal structure.

    Returns:
    - blocks: A list of dense matrices representing the diagonal blocks.
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

    Parameters:
    - blocks: A list of dense matrices.

    Returns:
    - inverted_blocks: A list of inverted dense matrices.
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

    Parameters:
    - inverted_blocks: A list of inverted dense matrices.

    Returns:
    - A sparse matrix with the inverted blocks on its diagonal.
    """
    # Use scipy.sparse.block_diag to create a block diagonal sparse matrix
    sparse_block_diag = sp.block_diag(inverted_blocks)
    return sparse_block_diag


def invert_diag_block_matrix(sparse_matrix, maxsize=3000):
    """
    Extract diagonal blocks from a sparse block diagonal matrix, compute the inverse of each block,
    and assemble the inverted blocks into a sparse block diagonal matrix.

    Parameters:
    - sparse_matrix: A sparse matrix (CSR format) with a block diagonal structure.
    - maxsize: The maximum size of the block diagonal matrix will be operated in a dense format.

    Returns:
    - sparse_inverted_block_matrix: A sparse/dense matrix with the inverted blocks on its diagonal.
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
    Compute the cholesky decomposition A = L * L' of a sparse symmetric, positive-definite matrix

    Parameters:
    - sparse_matrix: A diagonal sparse matrix in CSR or CSC format.
    - maxsize: The maximum size of the block diagonal matrix will be operated in a dense format.

    Returns:
    - sparse_matrix_inv: The cholesky decomposition factor L of the input sparse matrix.
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
