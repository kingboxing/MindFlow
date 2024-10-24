# FERePack/src/LinAlg/__init__.py
"""
The `LinAlg` subpackage provides linear algebra utilities and operations for use in finite element simulations
and other numerical computations. It includes functions for matrix and vector operations, assembling finite element matrices,
solving linear systems, and generating common vectors for simulations.

Modules and Classes
-------------------

- **MatrixOps**:
    - Provides classes and functions for assembling matrices, converting between different matrix formats,
      and performing matrix operations relevant to finite element analysis.
    - **Classes**:
        - `SparseLUSolver`: Base class for sparse LU solvers.
        - `FEniCSLU`: LU solver using FEniCS functions with PETSc backend.
        - `PETScLU`: LU solver using PETSc in petsc4py.
        - `SuperLU`: LU solver using SuperLU in scipy.sparse.linalg.
        - `UmfpackLU`: LU solver using UMFPACK in scikits.umfpack.
        - `InverseMatrixOperator`: Inverse operator for iterative solvers requiring matrix-vector products \( A^{-1} \mathbf{b} \).
    - **Functions**:
        - `ConvertMatrix`: Convert between FEniCS PETScMatrix and scipy.sparse.csr_matrix.
        - `ConvertVector`: Convert between FEniCS PETScVector and NumPy array.
        - `AssembleSystem`: Assemble a linear system from FEniCS expressions.
        - `AssembleMatrix`: Assemble a matrix from a UFL expression.
        - `AssembleVector`: Assemble a vector from a UFL expression.
        - `TransposePETScMat`: Transpose a FEniCS PETScMatrix.

- **MatrixAsm**:
    - Contains functions for constructing prolongation matrices, mass matrices, and identity matrices with boundary
      or subdomain conditions.
    - **Functions**:
        - `IdentMatProl`: Construct the prolongation matrix that excludes specified subspaces.
        - `MatWgt`: Assemble the mass (weight) matrix for the entire function space.
        - `IdentMatBC`: Construct an identity matrix with zeros at DOFs corresponding to boundary conditions.
        - `IdentMatSub`: Construct an identity matrix with zeros outside a specified subdomain.
        - `MatP`: Construct a prolongation matrix for resolvent analysis, excluding pressure subspaces.
        - `MatM`: Assemble the mass matrix restricted to the velocity subspace for resolvent analysis.
        - `MatQ`: Assemble the mass matrix restricted to the velocity subspace without boundary conditions.
        - `MatD`: Construct a prolongation matrix for a specified subdomain.

- **Utils**:
    - Contains utility functions for linear algebra operations, sparse matrix manipulations, and helper functions
      for working with FEniCS functions and data structures.
    - **Functions**:
        - `assign2`: Assign values from one FEniCS function to another.
        - `eigen_decompose`: Perform eigenvalue decomposition.
        - `assemble_sparse`: Assemble block matrices into a single sparse matrix.
        - `assemble_dae2`: Assemble state-space matrices for a DAE2 system.
        - `is_diag_sparse`: Check if a sparse matrix is diagonal.
        - `find_block_boundaries`: Find block boundaries in a sparse block diagonal matrix.
        - `extract_diagonal_blocks`: Extract diagonal blocks from a sparse block diagonal matrix.
        - `invert_dense_blocks`: Compute the inverse of each dense block matrix.
        - `assemble_diag_block_matrix`: Assemble inverted blocks into a sparse block diagonal matrix.
        - `invert_diag_block_matrix`: Invert a sparse block diagonal matrix.
        - `cholesky_sparse`: Compute the Cholesky decomposition of a sparse symmetric positive-definite matrix.
        - `woodbury_solver`: Solve a system using the Woodbury matrix identity.
        - `find_orthogonal_complement`: Find an orthogonal basis that complements a given subspace.

- **VectorAsm**:
    - Provides classes for generating common input/output vectors for solvers, particularly useful in frequency
      response analysis and fluid dynamics simulations.
    - **Classes**:
        - `VectorGenerator`: A factory class to generate common input and output vectors for simulations.

Usage
-----
To utilize the utilities provided by the `LinAlg` subpackage, you can import the necessary functions or modules as follows:

```python
from FERePack.LinAlg.Utils import assign2, eigen_decompose
from FERePack.LinAlg.MatrixOps import AssembleMatrix, InverseMatrixOperator
from FERePack.LinAlg.MatrixAsm import MatWgt, MatP
from FERePack.LinAlg.VectorAsm import VectorGenerator
```
Notes
-----
- **Dependencies**: Ensure that the required dependencies such as NumPy, SciPy, and FEniCS are installed and properly configured.
- **Sparse Matrices**: Many functions are designed to work with sparse matrices in CSR or CSC formats.
- **Finite Element Support**: Functions are specifically designed to work with FEniCS function objects and finite element spaces.

Examples
--------

Example of performing an eigenvalue decomposition:

```python
from FERePack.LinAlg.Utils import eigen_decompose
import scipy.sparse as sp

# Create sparse matrices A and M
A = sp.csr_matrix(...)
M = sp.csr_matrix(...)

# Perform eigenvalue decomposition
vals, vecs = eigen_decompose(A, M, k=5, sigma=0.1)
```
Example of assembling a matrix and solving a linear system:

```python
from FERePack.LinAlg.MatrixOps import AssembleMatrix, InverseMatrixOperator
from dolfin import *

# Define the variational formulation
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Assemble the matrix
A = AssembleMatrix(a)

# Create an inverse operator
invA = InverseMatrixOperator(A)

# Assemble the right-hand side vector
b = AssembleVector(L)

# Solve the linear system
x = invA @ b
```
Example of generating a Gaussian vector:

```python
from FERePack.LinAlg.VectorAsm import VectorGenerator

# Initialize the vector generator
vector_gen = VectorGenerator(element, bc_obj)

# Generate a Gaussian vector
gaussian_vec = vector_gen.gaussian_vector(center=(0.5, 0.5), sigma=0.1, scale=1.0, index=0)
```
"""

from .MatrixOps import *
from .MatrixAsm import *
from .Utils import *
from .VectorAsm import *