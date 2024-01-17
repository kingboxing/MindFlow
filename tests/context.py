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
from dolfin import *
import numpy as np
import time

# MindFlow pkgs
from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from src.NSolver.Steady.SteadyNewtonSolver import NewtonSolver