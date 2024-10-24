#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides functions for interfacing Python with MATLAB scripts using the MATLAB Engine API for Python.

It includes utilities for starting the MATLAB engine, adding paths to MATLAB, converting between Python and MATLAB
data types (including sparse matrices), and passing data structures between Python and MATLAB.

Functions
---------
- `start_matlab()`:
    Start the MATLAB engine and add necessary paths.
- `add_matlab_path(eng, mod_list=[])`:
    Add the current folder and specified module folders to the MATLAB path.
- `spmat2dict(spmat)`:
    Convert a SciPy sparse matrix to a dictionary suitable for MATLAB.
- `dict2spmat(matrix_dict)`:
    Reconstruct a SciPy sparse matrix from a dictionary received from MATLAB.
- `python2matlab(data)`:
    Convert Python data types to MATLAB-compatible types.
- `matlab2python(data)`:
    Convert MATLAB data types to Python-compatible types.
- `convert_to_matlab_dtype(python_dict)`:
    Convert a Python dictionary to MATLAB-compatible data types recursively.
- `convert_to_matlab_struct(eng, python_dict, struct_name='struct_var')`:
    Pass a Python dictionary to MATLAB and construct a MATLAB struct.

Dependencies
------------
- NumPy
- SciPy
- MATLAB Engine API for Python (`matlab.engine`)
- OS module

Ensure that the MATLAB Engine API for Python is installed and properly configured.

Examples
--------
Typical usage involves starting the MATLAB engine, converting data types, and passing data between Python and MATLAB.

Notes
-----
- This module is part of the 'Interface' subpackage of FERePack.
- Be cautious with data types when converting between Python and MATLAB, especially with sparse matrices and complex numbers.

"""

from ..Deps import *
import os


#%%
def start_matlab():
    """
    Start the MATLAB engine and add necessary paths.

    Returns
    -------
    eng : matlab.engine.MatlabEngine
        An instance of the MATLAB engine.

    Notes
    -----
    - This function starts the MATLAB engine and automatically adds the current directory and the 'Py2Matlab' subdirectory to the MATLAB path.
    - Ensure that the MATLAB Engine API for Python is installed and configured properly.
    """
    eng = matlab.engine.start_matlab()
    add_matlab_path(eng)
    return eng


def add_matlab_path(eng, mod_list=[]):
    """
    Add the current folder, subfolder 'Py2Matlab', and specified module folders to the MATLAB path.

    Parameters
    ----------
    eng : matlab.engine.MatlabEngine
        The MATLAB engine instance.
    mod_list : list of str, optional
        A list of folder names (modules) that should be added to the MATLAB path.
        These folders should be in the parent directory of the current script.
        Default is an empty list.

    Notes
    -----
    - This function modifies the MATLAB path within the MATLAB engine session.
    - It adds the current directory, the 'Py2Matlab' subdirectory, and any additional modules specified in `mod_list`.
    """
    path_list = []
    # Get the directory of the current Python script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Get the parent directory of the current script
    parent_dir = os.path.dirname(current_dir)
    # Add the current folder to the MATLAB path
    eng.addpath(current_dir, nargout=0)

    child_dir = os.path.join(current_dir, 'Py2Matlab')
    eng.addpath(child_dir, nargout=0)

    for mod in mod_list:
        mod_path = os.path.join(parent_dir, mod)  # Construct the full path
        eng.addpath(mod_path, nargout=0)
        path_list.append(mod_path)

    path_list.append(current_dir)
    path_list.append(child_dir)
    #return path_list


def spmat2dict(spmat):
    """
    Convert a SciPy sparse matrix to a dictionary suitable for MATLAB.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        The SciPy sparse matrix (CSR, CSC, COO, etc.) to be converted.

    Returns
    -------
    matrix_dict : dict
        A dictionary containing the data, rows, cols, shape, and format of the sparse matrix.

    Notes
    -----
    - The returned dictionary is compatible with MATLAB and can be converted back using `dict2spmat`.
    - Indices are adjusted to be 1-based to match MATLAB's indexing.
    - MATLAB only accepts double (float/complex) or logical data; integer types are converted to float.
    """
    if not sp.issparse(spmat):
        raise ValueError("Input is not a valid SciPy sparse matrix.")

    rows, cols = spmat.nonzero()
    data = spmat.data
    if np.issubdtype(data.dtype, (int, np.integer)):
        # matlab only accept double (float/complex) or logical data, no int
        data = data.astype(float)

    matrix_dict = {
        'type': 'sparse matrix',
        'format': spmat.getformat(),  # 'csr', 'csc', 'coo', etc.
        'data': data,  # Non-zero values
        'shape': np.array(spmat.shape),  # Shape of the matrix
        'rows': rows + 1,  # for matlab compatibility
        'cols': cols + 1,  # for matlab compatibility
    }

    return matrix_dict


def dict2spmat(matrix_dict):
    """
    Reconstruct a SciPy sparse matrix from a dictionary containing reconstruction information.

    Parameters
    ----------
    matrix_dict : dict
        The dictionary containing the sparse matrix data, as produced by `spmat2dict`.

    Returns
    -------
    spmatrix : scipy.sparse.spmatrix
        A SciPy sparse matrix (CSR, CSC, or COO) reconstructed from the dictionary.

    Notes
    -----
    - Indices are adjusted back to 0-based indexing for Python.
    - Supported formats are 'csr', 'csc', and 'coo'.
    - Raises a ValueError if the format is unsupported.
    """
    # Extract relevant information from the dictionary
    format = matrix_dict['format']
    data = matrix_dict['data'].flatten()
    shape = matrix_dict['shape'].astype(int).flatten()

    # Convert 1-based indexing (MATLAB) back to 0-based indexing (Python)
    rows = matrix_dict['rows'].astype(int).flatten() - 1
    cols = matrix_dict['cols'].astype(int).flatten() - 1

    # Reconstruct the sparse matrix based on its format
    if format == 'csr':
        return sp.csr_matrix((data, (rows, cols)), shape=shape)
    elif format == 'csc':
        return sp.csc_matrix((data, (rows, cols)), shape=shape)
    elif format == 'coo':
        return sp.coo_matrix((data, (rows, cols)), shape=shape)
    else:
        raise ValueError(f"Unsupported format: {format}")


def python2matlab(data):
    """
    Convert common Python data types to MATLAB-compatible types using the MATLAB Engine API.

    Parameters
    ----------
    data : any
        The Python data to be converted (int, float, list, tuple, dict, numpy array, etc.).

    Returns
    -------
    matlab_data : MATLAB data type
        MATLAB-compatible data (e.g., matlab.double, matlab.int64, etc.).

    Notes
    -----
    - Supports conversion of NumPy arrays, scalars, lists, tuples, dictionaries, and SciPy sparse matrices.
    - For unknown or unsupported types, raises a ValueError.
    - Be cautious with data types and ensure compatibility with MATLAB functions.
    """
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.complexfloating):
        return complex(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):  # seems auto convert
        if np.issubdtype(data.dtype, np.integer):
            return matlab.int64(data.tolist())
        elif np.issubdtype(data.dtype, np.floating):
            return matlab.double(data.tolist())
        elif np.issubdtype(data.dtype, np.complexfloating):
            return matlab.double(data.tolist(), is_complex=True)
        elif np.issubdtype(data.dtype, np.bool_):
            return matlab.logical(data.tolist())
        else:
            return data  #.tolist() # auto convert?
    elif isinstance(data, dict):
        matlab_struct = {}
        for key, value in data.items():
            matlab_struct[key] = python2matlab(value)
        return matlab_struct
    elif sp.issparse(data):
        sp_dict = spmat2dict(data)
        return python2matlab(sp_dict)
    elif isinstance(data, (list, tuple, int, float, complex, str, bool)):
        return data
    else:
        raise ValueError(f"Unsupported type: {type(data)}")


def matlab2python(data):
    """
    Convert MATLAB data types to Python-compatible data types.

    Parameters
    ----------
    data : any
        The MATLAB data to be converted (matlab.double, matlab.int64, etc.).

    Returns
    -------
    python_data : any
        Python-compatible data (int, float, numpy array, dict, etc.).

    Notes
    -----
    - Supports conversion of MATLAB arrays, structures, and primitive types.
    - For unknown or unsupported types, raises a ValueError.
    - Be cautious when handling complex numbers and sparse matrices.
    """
    if isinstance(data, matlab.double):
        # Handle real and complex double arrays
        arr = np.array(data)
        if data._is_complex:  # Check if it's complex
            return arr.astype(complex)
        return arr
    elif isinstance(data, matlab.int64):
        # Convert MATLAB int64 to NumPy integer array
        return np.array(data, dtype=np.int64)
    elif isinstance(data, matlab.logical):
        # Convert MATLAB logical to NumPy boolean array
        return np.array(data, dtype=bool)
    elif isinstance(data, dict):
        python_dict = {}
        for key, value in data.items():
            python_dict[key] = matlab2python(value)
        if data.get('type') == 'sparse matrix':
            return dict2spmat(python_dict)
        else:
            return python_dict
    elif isinstance(data, list):
        # Convert MATLAB cell array (list) to Python list
        return [matlab2python(item) for item in data]
    elif isinstance(data, (int, float, complex, bool, str)):
        # Direct conversion for primitive types
        return data
    else:
        raise ValueError(f"Unsupported type: {type(data)}")


#%%
def convert_to_matlab_dtype(python_dict):
    """
    Recursively convert a Python dictionary to a dictionary with data types acceptable by MATLAB.
    e.g.    list of numbers
            tuple of numbers
            int, float, np.integer, np.floating
            string
            numpy ndarray
            dict
            scipy.sparse matrix

    Parameters
    ----------
    python_dict : dict
        The Python dictionary to be converted.

    Returns
    -------
    matlab_dict : dict
        A dictionary containing data compatible with MATLAB data types.

    Notes
    -----
    - This function is intended to prepare data for passing to MATLAB functions via the MATLAB Engine.
    - Supports basic data types, NumPy arrays, SciPy sparse matrices, and nested dictionaries.
    - Converts data to types such as matlab.double, matlab.int64, etc.
    """
    matlab_dict = {}
    for key, value in python_dict.items():
        # Convert Python values to MATLAB types
        if isinstance(value, list):  # If it's a list, convert it to matlab.double
            value = matlab.double(value)
        elif isinstance(value, tuple):
            value = matlab.double(list(value))
        elif isinstance(value, (int, float, np.integer, np.floating)):  # If it's a number, convert it to a double
            value = float(value)
        elif isinstance(value, str):
            value = f"'{value}'"
        elif isinstance(value, bool):
            value = matlab.logical(value)
        elif isinstance(value, np.ndarray):
            value = matlab.double(value.tolist())
        elif isinstance(value, complex):
            value = matlab.double(value, is_complex=True)
        elif isinstance(value, dict):
            value = convert_to_matlab_dtype(value)
        elif sp.issparse(value):
            value = spmat2dict(value)
            value = convert_to_matlab_dtype(value)
        else:
            raise TypeError('Containing Invalid datatype that cannot be converted to Matlab datatype')

        matlab_dict[key] = value

    return matlab_dict


def convert_to_matlab_struct(eng, python_dict, struct_name='struct_var'):
    """
    Recursively pass a Python dictionary (including nested dictionaries) to MATLAB and construct a MATLAB struct.

    Parameters
    ----------
    eng : matlab.engine.MatlabEngine
        The running MATLAB engine.
    python_dict : dict
        The Python dictionary to be passed to MATLAB (after conversion with `convert_to_matlab_dtype`).
    struct_name : str, optional
        The name of the MATLAB struct to create in the MATLAB workspace. Default is 'struct_var'.

    Returns
    -------
    None

    Notes
    -----
    - This function uses MATLAB's `eval` function to create and populate the struct.
    - For nested dictionaries, the function recursively constructs nested structs.
    - Be cautious with the use of `eval` and ensure that keys are valid MATLAB struct field names.
    """

    for key, value in python_dict.items():
        if isinstance(value, dict):
            # Create a new struct field in MATLAB and recursively fill it
            eng.eval(f'{struct_name}.{key} = struct();', nargout=0)
            # Recursively call the function to handle the nested dictionary
            convert_to_matlab_struct(eng, value, struct_name=f'{struct_name}.{key}')
        else:
            # Dynamically assign the value to the MATLAB struct
            eng.eval(f'{struct_name}.{key} = {value};', nargout=0)
