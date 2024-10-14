"""
python interface to Matlab script
"""

from ..Deps import *
import os

#%%
def start_matlab():
    """
    start Matlab backend
    """
    eng = matlab.engine.start_matlab()
    add_matlab_path(eng)
    return eng

def add_matlab_path(eng, mod_list=[]):
    """
    Adds the current folder, subfolder 'Py2Mat' and specified module folders to the MATLAB path.
    Parameters:
    - eng: The MATLAB engine instance.
    - mod_list: A list of folder names (modules) that should be added to the MATLAB path.
                These folders should be in the same directory as the current script.
    """
    path_list=[]
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
    Convert a SciPy sparse matrix to a dictionary containing the information needed to reconstruct it in Matlab.

    Parameters:
    - spmat: The SciPy sparse matrix (CSR, CSC, COO, etc.) to be converted.

    Returns:
    - A dictionary containing the data, rows, cols, shape, and format of the sparse matrix.
    """
    if not sp.issparse(spmat):
        raise ValueError("Input is not a valid SciPy sparse matrix.")

    rows, cols = spmat.nonzero()
    data = spmat.data
    if np.issubdtype(data.dtype, (int, np.integer)):
        # matlab only accept double (float/complex) or logical data, no int
        data = data.astype(float)

    matrix_dict = {
        'type':     'sparse matrix',
        'format':   spmat.getformat(),  # 'csr', 'csc', 'coo', etc.
        'data':     data,  # Non-zero values
        'shape':    np.array(spmat.shape),  # Shape of the matrix
        'rows':     rows + 1,  # for matlab compatibility
        'cols':     cols + 1,  # for matlab compatibility
    }

    return matrix_dict


def dict2spmat(matrix_dict):
    """
    Reconstruct a SciPy sparse matrix from a dictionary containing the reconstruction information.

    Parameters:
    - matrix_dict: The dictionary containing the sparse matrix data, as produced by spmat2dict.

    Returns:
    - A SciPy sparse matrix (CSR, CSC, or COO).
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
    Convert common Python data types to MATLAB-compatible types using MATLAB Engine API.

    Parameters:
    - data: The Python data to be converted (int, float, list, tuple, dict, numpy array, etc.).

    Returns:
    - MATLAB-compatible data (matlab.double, matlab.int32, etc.).
    """
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.complexfloating):
        return complex(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray): # seems auto convert
        if np.issubdtype(data.dtype, np.integer):
            return matlab.int64(data.tolist())
        elif np.issubdtype(data.dtype, np.floating):
            return matlab.double(data.tolist())
        elif np.issubdtype(data.dtype, np.complexfloating):
            return matlab.double(data.tolist(), is_complex=True)
        elif np.issubdtype(data.dtype, np.bool_):
            return matlab.logical(data.tolist())
        else:
            return data #.tolist() # auto convert?
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
    Convert MATLAB-compatible types to common Python data types.

    Parameters:
    - data: The MATLAB data to be converted (matlab.double, matlab.int64, etc.).

    Returns:
    - Python-compatible data (int, float, numpy array, dict, etc.).
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
    Recursively Converts python dictionary to dictionary with datatypes acceptable by Matlab
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
        The Python dictionary to be passed to MATLAB.
    Returns
    -------
    - A dictionary containing the data that are compatible with Matlab datatypes.
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
    Recursively pass a Python dictionary (including nested dictionaries) to MATLAB
    and construct a corresponding MATLAB structure dynamically.

    Parameters
    ----------
    eng : MATLAB engine instance
        The running MATLAB engine.
    python_dict : dict
        The Python dictionary to be passed to MATLAB (returned by convert_to_matlab_dtype).
    struct_name : str, optional
        The name of the MATLAB structure to create in the MATLAB workspace.

    Returns
    -------
    None
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
