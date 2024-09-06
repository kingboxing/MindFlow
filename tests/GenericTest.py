import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
import scipy.linalg as sla
n=100

# Define the 5x5 matrix A
A = np.random.rand(n, n)
A[n-1, 0:n-1]=0
A[n-1, n-1]=1

A[0:n-1, n-1]=0
A[n-1, n-1]=0
# Define the 5x5 matrix B (must be symmetric and positive definite for generalized eigenvalue problems)
B = np.random.rand(n, n)
B = (B + B.T) / 2  # Make B symmetric
B += n * np.eye(n)  # Make B positive definite by adding a multiple of the identity
B[n-1, 0:n-1]=0
B[n-1, n-1]=1

B[0:n-1, n-1]=0
B[n-1, n-1]=1

# Convert matrices A and B to sparse format (scipy sparse matrices)
A_sparse = csc_matrix(A.astype(float))
B_sparse = csc_matrix(B.astype(float))
#C= np.linalg.inv(A)
#C_sparse = csc_matrix(C.astype(float))
# Solve the generalized eigenvalue problem A v = λ B v
eigenvalues, eigenvectors = eigs(A_sparse, M=B_sparse, k=1, which='LM')  # k=3 means we want 3 eigenvalues

# Display the eigenvalues and eigenvectors
print("Eigenvalues:")
print(eigenvalues)

vals,_=sla.eig(A,B)

print("All Eigenvalues:")
print(vals)
# print("Eigenvectors:")
# print(eigenvectors)