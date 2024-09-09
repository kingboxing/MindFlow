from dolfin import *
from petsc4py import PETSc

def create_identity_petsc_matrix_with_preallocation(n):
    """
    Create an identity PETScMatrix of size n x n using petsc4py and wrap it in a FEniCS PETScMatrix.
    Preallocate space for diagonal elements to handle boundary conditions.
    
    Parameters:
    - n: Size of the identity matrix (n x n)

    Returns:
    - fenics_matrix: A FEniCS PETScMatrix containing the identity matrix.
    """
    # Step 1: Create an empty PETSc matrix of size n x n
    A_petsc = PETSc.Mat().create()

    # Set the matrix size and type
    A_petsc.setSizes([n, n])
    A_petsc.setType("aij")  # Sparse AIJ format (compressed row storage)

    # Step 2: Preallocate space for the matrix
    # Since this is an identity matrix, we need 1 non-zero entry per row (on the diagonal)
    A_petsc.setPreallocationNNZ(30)  # 1 non-zero entry per row (for diagonal)

    # Step 3: Set the diagonal elements to 1 manually
    for i in range(n):
        A_petsc.setValue(i, i, 1.0)  # Set the diagonal element A[i, i] = 1

    # Step 4: Assemble the PETSc matrix
    A_petsc.assemble()

    # Step 5: Wrap the PETSc matrix in a FEniCS PETScMatrix
    fenics_matrix = PETScMatrix(A_petsc)

    return fenics_matrix

