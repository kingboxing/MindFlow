#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:51:37 2024

@author: bojin


"""
from src.Deps import *
from src.OptimControl.RiccatiSolver import GRiccatiDAE2Solver
try:
    import pymess as mess
except ImportError:
    MESS = False
    
class H2ControlSolver:
    """
    for LQG problems

    """
    def __init__(self, ssmodel):
        """
        | model.M   0 | d(|vel|     = | model.A   model.G  | |vel| + | model.B | u + | Bd | d
        |    0      0 |   |pre|)/dt   | model.GT model.Z=0 | |pre|   |    0    |     |  0 |
        
        y = | model.C   0 | |vel| + | Dn | n
                            |pre|            
        
        z = | Cs 0 | |vel| + | 0 | u
            | 0  0 | |pre|   | Du|
        
        
        model: state space model of linearised N-S equations
               model.A:  a block of state matrix
               model.G:  a block of state matrix
               model.GT: a block of state matrix
               model.Z:  a block of state matrix
               model.B:  input vector
               model.C:  output vector
               model.M:  full rank block of mass matrix
    
        k0: initial feedback that stabilises the system    
        
        Bd: weight matrix of disturbance d
        
        Cs: weight matrix of measured state for the cost function
        
        Du: weight matrix of control signal for the cost function
        
        Dn: weight matrix of sensor noise n
        
        option: options of eigen-decomposition for computing initial feedback
        """
        pass