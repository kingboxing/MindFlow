# FERePack/src/Interface/__init__.py
"""
The `Interface` subpackage provides interfaces for integrating Python with external software such as MATLAB, OpenFOAM, and ANSYS Fluent. It includes modules and functions to facilitate communication between Python and these applications, enabling data exchange and co-simulation in computational simulations and engineering analyses.

Modules and Functions
---------------------

- **Py2Mat**:
    - Provides functions for interfacing Python with MATLAB using the MATLAB Engine API for Python.
    - **Functions**:
        - `start_matlab()`: Start the MATLAB engine and add necessary paths.
        - `add_matlab_path(eng, mod_list=[])`: Add the current folder and specified module folders to the MATLAB path.
        - `spmat2dict(spmat)`: Convert a SciPy sparse matrix to a dictionary suitable for MATLAB.
        - `dict2spmat(matrix_dict)`: Reconstruct a SciPy sparse matrix from a dictionary received from MATLAB.
        - `python2matlab(data)`: Convert Python data types to MATLAB-compatible types.
        - `matlab2python(data)`: Convert MATLAB data types to Python-compatible types.
        - `convert_to_matlab_dtype(python_dict)`: Convert a Python dictionary to MATLAB-compatible data types recursively.
        - `convert_to_matlab_struct(eng, python_dict, struct_name='struct_var')`: Pass a Python dictionary to MATLAB and construct a MATLAB struct.

Usage
-----

To utilize the utilities provided by the `Interface` subpackage, you can import the necessary functions or modules as follows:

```python
from FERePack.Interface.Py2Mat import start_matlab, python2matlab, matlab2python
```

Notes
-----

- **Dependencies**: Ensure that the MATLAB Engine API for Python is installed and properly configured.
- **Data Conversion**: Be cautious with data types when converting between Python and MATLAB, especially with sparse matrices and complex numbers.
- **External Software**: Additional modules for OpenFOAM and Fluent interfaces are planned but may not be fully implemented yet.

Examples
-----

Example of starting the MATLAB engine and converting data between Python and MATLAB:

```python
import scipy.sparse as sp
from FERePack.Interface.Py2Mat import start_matlab, python2matlab, matlab2python

# Start the MATLAB engine
eng = start_matlab()

# Define Python data
python_data = {
    'array': [1, 2, 3],
    'matrix': [[1, 2], [3, 4]],
    'sparse_matrix': sp.csr_matrix([[0, 1], [2, 0]]),
    'complex_number': complex(1, 2),
}

# Convert Python data to MATLAB-compatible types
matlab_data = python2matlab(python_data)

# Pass data to MATLAB workspace
eng.workspace['data'] = matlab_data

# Perform operations in MATLAB
eng.eval('result = data.matrix * data.array\';', nargout=0)

# Retrieve result from MATLAB
result = eng.workspace['result']
python_result = matlab2python(result)

print("Result from MATLAB:", python_result)
```

In this example:

- We start the MATLAB engine using start_matlab().
- Define a Python dictionary containing various data types, including a sparse matrix and a complex number.
- Convert the Python data to MATLAB-compatible types using python2matlab().
- Pass the data to the MATLAB workspace and perform matrix operations.
- Retrieve the result from MATLAB and convert it back to Python types using matlab2python().
"""

from .Py2Mat import *