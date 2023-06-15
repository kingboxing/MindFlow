from __future__ import print_function
from fenics import *
import numpy as np
from scipy.sparse import csr_matrix
from petsc4py import PETSc
"""This module provides classes that convert FEniCS expression into CSR matrix/vector
"""


class MatrixAssemble:
    def __init__(self):
        pass

    def assemblesystem(self, mat_expression, vec_expression, bcs=[]):
        A= PETScMatrix()
        b = PETScVector()
        assemble_system(mat_expression, vec_expression, bcs, A_tensor=A, b_tensor=b)
        A_sparry=self.convertmatrix(A)
        b_vec=self.convertvector(b)
        return A_sparry, b_vec

    def assemblematrix(self,expression,bcs=[]):
        """General function to assemble the matrix and convert it to CSR format

        Parameters
        ----------------------------
        expression : UFL expression

        bcs : list
            boundary conditions that change values in the specified rows

        Returns
        ----------------------------
        A_sparray : sparse matrix in CSR format

        """
        A = PETScMatrix() # FEniCS using PETSc for matrix operation
        assemble(expression, tensor=A) # store assembled matrix in A
        [bc.apply(A) for bc in bcs]
        # convert the format to CSR
        A_mat = as_backend_type(A).mat()
        A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)
        return A_sparray

    def convertmatrix(self, Matrix,flag='PETSc2Mat'):
        if flag=='PETSc2Mat':
            A_mat = as_backend_type(Matrix).mat()
            A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)
            return A_sparray
        elif flag=='Mat2PETSc':
            A_mat = Matrix.tocsr()
            p1=A_mat.indptr
            p2=A_mat.indices
            p3=A_mat.data
            petsc_mat = PETSc.Mat().createAIJ(size=A_mat.shape,csr=(p1,p2,p3))# creating petsc matrix from csc
            A_petsc = PETScMatrix(petsc_mat)
            return A_petsc

    def assemblevector(self,expression,bcs=[]):
        """General function to assemble the vector and convert it to 1D matrix

        Parameters
        ----------------------------
        expression : UFL expression

        bcs : list
            boundary conditions that change values in the specified rows

        Returns
        ----------------------------
        A_vec : 1D matrix

        """
        A = PETScVector() # FEniCS using PETSc for matrix operation
        assemble(expression, tensor=A) # store assembled matrix in A
        [bc.apply(A) for bc in bcs]
        # convert the matrix
        A_vec = np.matrix(A.get_local())
        return A_vec

    def convertvector(self, vector,flag='PETSc2Vec'):
        if flag=='PETSc2Vec':
            A_vec = np.matrix(vector.get_local())
            return A_vec
        elif flag=='Vec2PETSc':
            A_vec=np.asarray(vector)
            return PETScVector(PETSc.Vec().createWithArray(A_vec))

            