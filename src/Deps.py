#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:35:12 2024

@author: bojin
"""

from dolfin import *
import numpy as np
import scipy.io as sio
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import copy
import gc
import multiprocessing
from petsc4py import PETSc
import joblib as jb

try:
    import matplotlib.pyplot as plt
    has_mpl = True
except ImportError:
    has_mpl = False

try:
    from scikits import umfpack
    has_umfpack = True
except ImportError:
    has_umfpack = False

try:
    from sksparse.cholmod import cholesky
    has_cholesky = True
except ImportError:
    has_cholesky = False

try:
    import pymess as mess
    has_pymess = True
except ImportError:
    has_pymess = False

try:
    import matlab.engine
    has_matlab = True
except ImportError:
    has_matlab = False

parameters["std_out_all_processes"] = False
comm_mpi4py = MPI.comm_world
comm_rank = comm_mpi4py.Get_rank()
comm_size = comm_mpi4py.Get_size()
