#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:24:05 2024

@author: bojin
"""

from dolfin import *

class NSolveParams:
    """
    
    """
    def __init__(self):
        self.params={}
        
    def NewtonSolver(self):
        self.params.update({'newton_solver':{'linear_solver': 'mumps','absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12}})
        
    
    
    

class FreqOptions:
    """
    """
    
    def __init__(self):
        pass
    

class ContOptions:
    """
    """
    
    def __init__(self):
        pass