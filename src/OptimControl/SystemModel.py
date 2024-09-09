#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 20:38:07 2024

@author: bojin
"""

from src.Deps import *

from src.FreqAnalys.FreqSolverBase import FreqencySolverBase
from src.LinAlg.MatrixOps import AssembleMatrix, AssembleVector, InverseMatrixOperator
from src.LinAlg.MatrixAsm import MatP, MatM, MatQ, MatD


class StateSpaceDAE2:
    def __init__(self):
        pass