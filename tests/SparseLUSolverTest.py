#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:08:19 2024

@author: bojin
"""

from context import *
from src.LinAlg.MatrixOps import FEniCSLU, PETScLU, SuperLU, UmfpackLU

Ar = sp.csr_matrix([[4, 1, 5], [1, 3, 2], [2,1,4]], dtype=np.float64)
Ai = sp.csr_matrix([[0, -1, -2], [1, 0, 4],[-1,3,5]], dtype=np.float64)
A = Ar + 1j * Ai
vr = np.array([1, 3, 4], dtype=np.float64)
vi = np.array([2, 4, 6], dtype=np.float64)
v = vr + 1j*vi


def test_MatInv(lusolver, trans, A, b):
    print("Testing InverseMatrixOperator("+lusolver+", trans="+trans+")...")
    # Create a small complex sparse matrix

    A_complex = A
    
    # Create a complex right-hand side vector
    b = b
    
    # Initialize the MatInv solver with the complex matrix
    # This example uses the 'mumps' solver, which is a common choice for LU factorization.
    mat_inv =InverseMatrixOperator(A_complex, lusolver=lusolver, trans=trans)
    
    # Solve the system A*x = b
    x = mat_inv.matvec(b)
    
    # Output the result
    print("Solution x:", x)
    
    
    # Verify by computing A*x and comparing it with b
    if trans == 'N':
        Ax = A_complex.dot(x)
    elif trans == 'T':
        Ax = A_complex.T.dot(x)
    elif trans == 'H':
        Ax = A_complex.T.conjugate().dot(x)
    
    print("b=Ax:", Ax)
    print("Original b:", b)
    assert np.allclose(Ax,b, atol=1e-6), "Ax does not match the original b."


def test_MatInvUV(lusolver, trans, A, b):
    print("Testing InverseMatrixOperator(" + lusolver + ", trans=" + trans + ")...")

    U = np.random.rand(3, 1) + np.random.rand(3, 1) * 1j
    V = np.random.rand(3, 1) + np.random.rand(3, 1) * 1j
    # b = np.random.rand(3) + np.random.rand(3) * 1j

    Mat = {'U': U, 'V': V}

    # Initialize the MatInv solver with the complex matrix
    # This example uses the 'mumps' solver, which is a common choice for LU factorization.
    mat_inv = InverseMatrixOperator(A, Mat=Mat, lusolver=lusolver, trans=trans)

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

    print("b=(A+UV')x:", Ax)
    print("Original b:", b)
    assert np.allclose(Ax, b, atol=1e-6), "Ax does not match the original b."

def test_SparseLU(func, trans, A, x):
    print("Testing "+func+"(trans="+trans+")...")
    # Step 1: Define a small complex sparse matrix A and a complex vector b
    # Here, we create a simple 2x2 system for testing purposes
    A = A
    
    # Create a known solution x
    x_known = x
    
    # Compute the RHS vector b = A * x_known
    if trans == 'N':
        b = A.dot(x_known)
    elif trans == 'T':
        b = A.T.dot(x_known)
    elif trans == 'H':
        b = A.T.conjugate().dot(x_known)
    
    # Step 2: Initialize the FEniCSLU solver with the complex matrix A
    solver = eval(func+'(A)')
    
    # Step 3: Solve the linear system A * x = b using FEniCSLU
    x_computed = solver.solve(b, trans)
    
    # Step 4: Check if the computed solution matches the known solution
    print("RHS:", b)
    print("Known solution:", x_known)
    print("Computed solution:", x_computed)
    
    # Assert that the computed solution is close to the known solution
    assert np.allclose(x_computed, x_known, atol=1e-6), "The solution does not match the expected result."

# Run the test
if __name__ == "__main__":
    print("***(A+UV')x=b (Complex LHS & complex RHS)****")
    test_MatInvUV('mumps', 'N', A, v)
    test_MatInvUV('mumps', 'T', A, v)
    test_MatInvUV('mumps', 'H', A, v)
    test_MatInvUV('petsc', 'N', A, v)
    test_MatInvUV('petsc', 'T', A, v)
    test_MatInvUV('petsc', 'H', A, v)
    test_MatInvUV('superlu', 'N', A, v)
    test_MatInvUV('superlu', 'T', A, v)
    test_MatInvUV('superlu', 'H', A, v)
    test_MatInvUV('umfpack', 'N', A, v)
    test_MatInvUV('umfpack', 'T', A, v)
    test_MatInvUV('umfpack', 'H', A, v)
    print("***Complex LHS & complex RHS****")
    test_MatInv('mumps', 'N', A, v)
    test_MatInv('mumps', 'T', A, v)
    test_MatInv('mumps', 'H', A, v)
    test_MatInv('petsc', 'N', A, v)
    test_MatInv('petsc', 'T', A, v)
    test_MatInv('petsc', 'H', A, v)
    test_MatInv('superlu', 'N', A, v)
    test_MatInv('superlu', 'T', A, v)
    test_MatInv('superlu', 'H', A, v)
    test_MatInv('umfpack', 'N', A, v)
    test_MatInv('umfpack', 'T', A, v)
    test_MatInv('umfpack', 'H', A, v)
    print("***Real LHS & complex RHS****")
    test_MatInv('mumps', 'N', Ar, v)
    test_MatInv('mumps', 'T', Ar, v)
    test_MatInv('mumps', 'H', Ar, v)
    test_MatInv('petsc', 'N', Ar, v)
    test_MatInv('petsc', 'T', Ar, v)
    test_MatInv('petsc', 'H', Ar, v)
    test_MatInv('superlu', 'N', Ar, v)
    test_MatInv('superlu', 'T', Ar, v)
    test_MatInv('superlu', 'H', Ar, v)
    test_MatInv('umfpack', 'N', Ar, v)
    test_MatInv('umfpack', 'T', Ar, v)
    test_MatInv('umfpack', 'H', Ar, v)
    print("***Complex LHS & real RHS****")
    test_MatInv('mumps', 'N', A, vr)
    test_MatInv('mumps', 'T', A, vr)
    test_MatInv('mumps', 'H', A, vr)
    test_MatInv('petsc', 'N', A, vr)
    test_MatInv('petsc', 'T', A, vr)
    test_MatInv('petsc', 'H', A, vr)
    test_MatInv('superlu', 'N', A, vr)
    test_MatInv('superlu', 'T', A, vr)
    test_MatInv('superlu', 'H', A, vr)
    test_MatInv('umfpack', 'N', A, vr)
    test_MatInv('umfpack', 'T', A, vr)
    test_MatInv('umfpack', 'H', A, vr)
    print("***Real LHS & real RHS****")
    test_MatInv('mumps', 'N', Ar, vr)
    test_MatInv('mumps', 'T', Ar, vr)
    test_MatInv('mumps', 'H', Ar, vr)
    test_MatInv('petsc', 'N', Ar, vr)
    test_MatInv('petsc', 'T', Ar, vr)
    test_MatInv('petsc', 'H', Ar, vr)
    test_MatInv('superlu', 'N', Ar, vr)
    test_MatInv('superlu', 'T', Ar, vr)
    test_MatInv('superlu', 'H', Ar, vr)
    test_MatInv('umfpack', 'N', Ar, vr)
    test_MatInv('umfpack', 'T', Ar, vr)
    test_MatInv('umfpack', 'H', Ar, vr)
    print("***LU testing: Complex LHS & complex RHS****")
    test_SparseLU('FEniCSLU','N', A, v)
    test_SparseLU('FEniCSLU','T', A, v)
    test_SparseLU('FEniCSLU','H', A, v)
    test_SparseLU('PETScLU','N', A, v)
    test_SparseLU('PETScLU','T', A, v)
    test_SparseLU('PETScLU','H', A, v)
    test_SparseLU('SuperLU','N', A, v)
    test_SparseLU('SuperLU','T', A, v)
    test_SparseLU('SuperLU','H', A, v)
    test_SparseLU('UmfpackLU','N', A, v)
    test_SparseLU('UmfpackLU','T', A, v)
    test_SparseLU('UmfpackLU','H', A, v)


