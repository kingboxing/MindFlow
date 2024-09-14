#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:19:50 2024

@author: bojin
"""

from src.Deps import *
from src.OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from src.OptimControl.BernoulliSolver import BernoulliFeedback
try:
    import pymess as mess
except ImportError:
    MESS = False
 
    
class LQRSolver(GRiccatiDAE2Solver):
    def __init__(self, ssmodel):
        """
        Solver for LQR problem of a Index 2 system

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__(self._initialise_model(ssmodel))
        self._k0=BernoulliFeedback(ssmodel)
        self.param['solver_type']='lqr_solver'
        self._default_param()
        
    def _initialise_model(self, ssmodel):
        """
        initialise model with necessary matrices

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            DESCRIPTION.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        model = {'A': ssmodel['A'],
                 'M': ssmodel['M'],
                 'G': ssmodel['G'],
                 'B': ssmodel['B'],
                 'C': ssmodel['C']}
        return model
    
    def Init_Feedback(self):
        param = {'nm.k0': self._k0.solve(transpose=False)}
        self.update_parameters(param)
        
        
    def _default_param(self):
        """
        initialise default parameters

        Returns
        -------
        None.

        """
        param_default = {'type': mess.MESS_OP_TRANSPOSE,
                         'adi.memory_usage': mess.MESS_MEMORY_HIGH,
                         'adi.paratype': mess.MESS_LRCFADI_PARA_ADAPTIVE_V,
                         'adi.output': 0,
                         'adi.res2_tol': 1e-8,
                         'adi.maxit': 2000,
                         'nm.output': 1,
                         'nm.singleshifts': 0,
                         'nm.linesearch': 1,
                         'nm.res2_tol': 1e-5,
                         'nm.maxit': 30,
                         #'nm.k0': None #Initial Feedback
                         }
        self.update_parameters(param_default)
        
    def ControlPenalty(self, beta):
        """
        magitude of the control penalty
        
        penalisation of the control signal: R^{-1/2} in the formulation
        ----------------------
        beta: int or float or 1D array
            magitude of the control penalty
        """
        n=self.Model['B'].shape[1]
        if isinstance(beta, (int, np.integer, float, np.floating)):
            mat = 1.0/beta * np.identity(n)
        elif isinstance(beta, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(beta[:n]))
        else:
            raise TypeError('Wrong type of control penalty.')
        
        self.Model['B_pen_mat'] = mat
        self.Model['B'] = self.Model['B'] @ mat 
    
    def Measure(self, Cz = None):
        """
        square root of response weight matrix: Q^{1/2}
        -----------------------
        Cz: 1d array or ndarray
        """
        
        if Cz is not None and Cz.shape[1] == self.Model['C'].shape[1]:
            self.Model['C'] = Cz
        else:
            raise ValueError('Wrong shape of measurement')
               
    def Regulator(self):
        """
        compute linear quadratic regulator from the solution of LQR problem

        Returns
        -------
        Kr : TYPE
            DESCRIPTION.

        """
        Kr = self.Model['B_pen_mat'] * self._B.transpose() * self.facZ * self.facZ.transpose()
        return Kr
    
    def iter_solve(self, num_iter = 2):
        pass
    