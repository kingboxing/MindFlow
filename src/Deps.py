#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:35:12 2024

@author: bojin
"""

from dolfin import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time


parameters["std_out_all_processes"] = False
comm_mpi4py = MPI.comm_world
comm_rank = comm_mpi4py.Get_rank()
comm_size = comm_mpi4py.Get_size()