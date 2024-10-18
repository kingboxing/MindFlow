from context import *
from src.LinAlg.Utils import find_orthogonal_complement


n, k, l = 1000, 10, 1
A = np.random.rand(n, k)
U = np.random.rand(n, l)

# Generate a sparse weight matrix M (n x n)

M = sp.diags(np.random.rand(n))
U = U @ np.linalg.inv(np.sqrt(U.T @ M @ U))

# Set the traction tolerance to a desired value, e.g., 1e-4
tolerance = 1e-4

# Find the new orthogonal basis with the sparse weight matrix M
new_basis = find_orthogonal_complement(A, U, M, tolerance=tolerance)

print("Shape of new basis:", new_basis.shape)