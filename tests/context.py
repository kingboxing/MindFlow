#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:12:47 2023

@author: bojin
"""
# system envs
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# python pkgs
from src.Deps import *

# MindFlow pkgs
from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition, BoundaryCondition, Boundary
from src.NSolver.SteadySolver import NewtonSolver
from src.NSolver.TransiSolver import DNS_IPCS
from src.FreqAnalys.FreqSolver import FrequencyResponse
from src.LinAlg.VectorAsm import VectorGenerator
from src.FreqAnalys.EigenSolver import EigenAnalysis
from src.FreqAnalys.ResolSolver import ResolventAnalysis
from src.OptimControl.SystemModel import StateSpaceDAE2
from src.OptimControl.BernoulliSolver import BernoulliFeedback
from src.OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from src.OptimControl.LQESolver import LQESolver
from src.OptimControl.LQRSolver import LQRSolver

import psutil
import time