#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:36:04 2024

@author: bojin
"""

from src.Deps import *

from src.BasicFunc.Boundary import SetBoundary
from src.Eqns.NavierStokes import Incompressible


class NSolverBase:
    """
    Solver Base of Navier-Stokes equations


    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> see test 'CylinderBaseFlow.py'

    """
    
    def __init__(self, mesh, element, Re, sourceterm, bodyforce):
        """
        Solver Base for N-S equations

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        element : TYPE
            DESCRIPTION.
        Re : TYPE
            DESCRIPTION.
        sourceterm : TYPE
            DESCRIPTION.
        bodyforce : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.element = element
        # store solution
        self.flow=self.element.w
        # boundary
        self.boundary=SetBoundary(mesh)
        self.set_boundary=self.boundary.set_boundary
        # NS equations
        self.eqn=Incompressible(self.element, self.boundary, Re, sourceterm=sourceterm, bodyforce=bodyforce)
        # parameters
        self.param={}
        
        
