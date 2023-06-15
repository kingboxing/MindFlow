from __future__ import print_function
from fenics import *
import numpy as np
import pickle
from scipy.sparse import dia_matrix, coo_matrix,csc_matrix,csr_matrix
import scipy.sparse.linalg as spla
from .FiniteElement import *

#from FrequencyResponse.MatrixAssemble import MatrixAssemble
#from Boundary.Set_BoundaryCondition import BoundaryCondition
"""This module provides some useful functions
"""



"""
Define some useful functions
"""
def precondition_ilu(A, useUmfpack=False):
    """
    :param A: matrix A
    :param useUmfpack:
    :return: inverse of the preconditioner
    """
    spla.use_solver(useUmfpack=useUmfpack)
    if useUmfpack is True:
        A = to_int64(A.astype('D'))
    M2 = spla.spilu(A)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator(A.shape, M_x)
    return M

def precondition_lu(A, useUmfpack=False):
    """
    :param A: matrix A
    :param useUmfpack:
    :return: inverse of the preconditioner
    """
    spla.use_solver(useUmfpack=useUmfpack)
    if useUmfpack is True:
        A = to_int64(A.astype('D'))
    M2 = spla.splu(A)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator(A.shape, M_x)
    return M

def precondition_jacobi(A, useUmfpack=False):
    """
    :param A: matrix A
    :param useUmfpack:
    :return: inverse of the preconditioner
    """
    data = np.reciprocal(A.diagonal())
    offsets = 0
    M = dia_matrix((data, offsets), shape=A.shape)
    return M.tocsc()

def to_int64(x,mtype='csc'):
    y = eval(mtype+'_matrix(x).copy()')
    y.indptr = y.indptr.astype(np.int64)
    y.indices = y.indices.astype(np.int64)
    return y
    
def rmse(predictions, targets):
    """Root-mean-square deviation between two arrays

    Parameters
    ----------------------------
    predictions : Predicted array

    targets : Obtained array

    Returns
    ----------------------------
    Root-mean-square deviation

    """
    return np.sqrt(((predictions - targets) ** 2).mean())
    
def DataConvert_mesh(mesh1,mesh2, path1, path2, etype = 'TaylorHood', time = 0.0, time_store = 0.0):
    """
    Data convert between different meshes: from mesh1 to mesh2
    :param mesh1:
    :param mesh2:
    :param path1:
    :param path2:
    :param etype:
    :return:
    """

    parameters["allow_extrapolation"] = True
    if etype is 'TaylorHood' and type(path1) is not dict and type(path2) is not dict:
        element_1 = TaylorHood(mesh=mesh1)
        w1 = element_1.w
        element_2 = TaylorHood(mesh=mesh2)
        timeseries_1 = TimeSeries(path1)
        timeseries_1.retrieve(w1.vector(), time)
        w2 = project(w1, element_2.functionspace, solver_type='gmres')
        timeseries_2 = TimeSeries(path2)
        timeseries_2.store(w2.vector(), time_store)
    if etype is 'SplitPV' and type(path1) is dict and type(path2) is dict:
        element_1 = SplitPV(mesh=mesh1)
        u1 = element_1.u
        p1 = element_1.p
        timeseries_v1 = TimeSeries(path1['Velocity'])
        timeseries_v1.retrieve(u1.vector(), time)
        timeseries_p1 = TimeSeries(path1['Pressure'])
        timeseries_p1.retrieve(p1.vector(), time)

        element_2 = SplitPV(mesh=mesh2)
        u2 = project(u1, element_2.functionspace_V, solver_type='gmres')
        p2 = project(p1, element_2.functionspace_Q, solver_type='gmres')
        timeseries_v2 = TimeSeries(path2['Velocity'])
        timeseries_v2.store(u2.vector(), time_store)
        timeseries_p2 = TimeSeries(path2['Pressure'])
        timeseries_p2.store(p2.vector(), time_store)

def DataConvert_element(mesh1,mesh2, path1, path2, etype = 'TaylorHood2SplitPV', time = 0.0, time_store = 0.0):
    parameters["allow_extrapolation"] = True
    if etype is 'TaylorHood2SplitPV' and type(path1) is not dict and type(path2) is dict:
        element_1 = TaylorHood(mesh=mesh1)
        element_2 = SplitPV(mesh=mesh2)
        timeseries_1 = TimeSeries(path1)
        timeseries_1.retrieve(element_1.w.vector(), time)
        u2 = project(element_1.u, element_2.functionspace_V, solver_type='gmres')
        p2 = project(element_1.p, element_2.functionspace_Q, solver_type='gmres')
        timeseries_v2 = TimeSeries(path2['Velocity'])
        timeseries_v2.store(u2.vector(), time_store)
        timeseries_p2 = TimeSeries(path2['Pressure'])
        timeseries_p2.store(p2.vector(), time_store)
    elif etype is 'SplitPV2TaylorHood' and type(path2) is not dict and type(path1) is dict:
        element_1 = SplitPV(mesh=mesh1)
        element_2 = TaylorHood(mesh=mesh2)
        timeseries_v1 = TimeSeries(path1['Velocity'])
        timeseries_v1.retrieve(element_1.u.vector(), time)
        timeseries_p1 = TimeSeries(path1['Pressure'])
        timeseries_p1.retrieve(element_1.p.vector(), time)
        u2 = project(element_1.u, element_2.functionspace.sub(0), solver_type='gmres')
        p2 = project(element_1.p, element_2.functionspace.sub(1), solver_type='gmres')
        element_2.u.vector()[:] = u2.vector()[:]
        element_2.p.vector()[:] = p2.vector()[:]
        timeseries_2 = TimeSeries(path2)
        timeseries_2.store(element_2.w.vector(), time_store)

def sort_complex(a):
    """
    sort a complex array based on the real part then the imaginary part
    """
    b = np.core.numeric.array(a, copy=True)
    b.sort()
    index_sort=[]
    for i in b:
        index_sort.append(list(a).index(i))
    return b[::-1], index_sort[::-1]        
        
def is_symmetric(m):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check
    
def pinvs(a, cond=None, rcond=None, return_rank=False, check_finite=True):
    """
    remains to be modeified using linear operator for sparse matrices
    
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.
    Calculate a generalized inverse of a matrix using its
    singular-value decomposition and including all 'large' singular
    values.
    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    cond, rcond : float or None
        Cutoff for 'small' singular values.
        Singular values smaller than ``rcond*largest_singular_value``
        are considered zero.
        If None or -1, suitable machine precision is used.
    return_rank : bool, optional
        if True, return the effective rank of the matrix
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix.  Returned if return_rank == True
    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    Examples
    --------
    >>> from scipy import linalg
    >>> a = np.random.randn(9, 6)
    >>> B = linalg.pinv2(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True
    """
    if check_finite:
        a = np.asarray_chkfinite(a)
        
    u, s, vh = spla.svd(a, full_matrices=False, check_finite=False)

    if rcond is not None:
        cond = rcond
    if cond in [None,-1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps

    rank = np.sum(s > cond * np.max(s))
    psigma_diag = 1.0 / s[: rank]

    B = np.transpose(np.conjugate(np.dot(u[:, : rank] *
                                         psigma_diag, vh[: rank])))

    if return_rank:
        return B, rank
    else:
        return B
        
def partial_refine(mesh,condition):
    """
    refine mesh under certain conditions
    ----------
    mesh: Mesh object
    
    condition: str
    """
    cell_markers = MeshFunction('bool',mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    for cell in cells(mesh):
        p = cell.midpoint()
        x=[p.x(), p.y(), p.z()]
        if eval(condition):
            cell_markers[cell] = True
    mesh_new = refine(mesh, cell_markers)
    return mesh_new
    
class complex2bode:
    def __init__(self,fre,array_com):
        self.f=np.asarray(fre)
        self.vals=np.asarray(array_com)
        self.mag=np.abs(array_com)
        self.mag_db=self.mag2db(self.mag)
        self.ang=self.phase(array_com)
        self.ang_deg=self.phase(array_com,deg=True)
        
    def mag2db(self,mag):

        return 20. * np.log10(mag)
        
    def phase(self,array_com,deg=False):
        ang=[]
        array=list(array_com)
        for i in range(np.size(array)):
            if i==0:
                ang.append(np.angle(array[i],deg=deg))
            else:
                ang.append(np.angle(self.rotate_vec(array[i],ang[i-1],deg),deg=deg)[0]+ang[i-1])
           
        if np.size(ang) != np.size(array):
            info('%e != %e' %(np.size(ang),np.size(array)))
            raise ValueError('Unequal size of array')
        return ang
        
    def rotate_vec(self,vec,ang,deg):
        if deg==True:
            angle=ang/180.0*np.pi
        elif deg==False:
            angle=ang
        a=np.asarray([np.real(vec),np.imag(vec)])
        T=np.zeros((2,2))
        T[0,0]=np.cos(angle)
        T[0,1]=np.sin(angle)
        T[1,0]=-np.sin(angle)
        T[1,1]=np.cos(angle)
        A=np.asarray(T*np.matrix(a).transpose())
        return A[0]+A[1]*1j


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def read_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj