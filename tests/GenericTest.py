from context import *
from src.LinAlg.Utils import woodbury_solver

Ar = sp.csr_matrix([[4, 1, 5], [1, 3, 2], [2, 1, 4]], dtype=np.float64)
Ai = sp.csr_matrix([[0, -1, -2], [1, 0, 4], [-1, 3, 5]], dtype=np.float64)
A = Ar + 1j * Ai
vr = np.array([1, 3, 4], dtype=np.float64)
vi = np.array([2, 4, 6], dtype=np.float64)
v = vr + 1j * vi

U = np.random.rand(3, 1) + np.random.rand(3, 1) * 1j
V = np.random.rand(3, 1) + np.random.rand(3, 1) * 1j
b = np.random.rand(3) + np.random.rand(3) * 1j
I = np.identity(3)

Mat = {'U': U, 'V': V}
trans = 'H'

# Initialize the MatInv solver with the complex matrix
# This example uses the 'mumps' solver, which is a common choice for LU factorization.
mat_inv = InverseMatrixOperator(A, Mat=Mat, lusolver='mumps', trans=trans)

# Solve the system A*x = b
x = mat_inv.matvec(b)

# Output the result
print("Solution x:", x)

# Verify by computing A*x and comparing it with b
if trans == 'N':
    A_complex = A + U @ V.T
    Ax = A_complex.dot(x)
elif trans == 'T':
    A_complex = A.T + U @ V.T
    Ax = A_complex.dot(x)
elif trans == 'H':
    A_complex = A.T.conjugate() + U @ V.T
    Ax = A_complex.dot(x)

print("b=Ax:", Ax)
print("Original b:", b)
assert np.allclose(Ax, b, atol=1e-6), "Ax does not match the original b."
