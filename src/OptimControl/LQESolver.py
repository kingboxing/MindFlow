#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:19:23 2024

@author: bojin
"""

from src.Deps import *
from src.OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from src.OptimControl.BernoulliSolver import BernoulliFeedback
try:
    import pymess as mess
except ImportError:
    MESS = False
 
class LQESolver(GRiccatiDAE2Solver):
    def __init__(self, ssmodel):
        """
        Solver for LQE problem of a Index 2 system

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
        self.param['solver_type']='lqe_solver'
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
        
        
    def _default_param(self):
        """
        initialise default parameters

        Returns
        -------
        None.

        """
        param_default = {'type': mess.MESS_OP_NONE,
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
            
    def Init_Feedback(self):
        param = {'nm.k0': self._k0.solve(transpose=True)}
        self.update_parameters(param)
        
    def SensorNoise(self, alpha):
        """
        magitude of the sensor noise
        
        modelled as uncorrelated zero-mean Gaussian white noise.
        
        noise weight matrix: V^{-1/2} in the formulation
        ----------------------
        alpha: int or float or 1D array
            magitude of the sensor noise
        """
        n=self.Model['C'].shape[0]
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            mat = 1.0/alpha * np.identity(n)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(alpha[:n]))
        else:
            raise TypeError('Wrong type of sensor noise.')
        
        self.Model['C_pen_mat'] = mat
        self.Model['C'] = mat @ self.Model['C'] 
        
    def Disturbance(self, Bd = None):
        """
        square root of random disturbance covariance matrix: W^{1/2}
        -----------------------
        Wd: 1d array or ndarray
        """
        
        if Bd is not None and Bd.shape[0] == self.Model['B'].shape[0]:
            self.Model['B'] = Bd
        else:
            raise ValueError('Wrong shape of disturbances')
                
    def Estimator(self):
        """
        compute estimator from the solution of LQE problem

        Returns
        -------
        Kf : TYPE
            DESCRIPTION.

        """
        Kf=self.facZ*(self.facZ.transpose()*self._C.transpose()* self.Model['C_pen_mat'])
        return Kf
    
    def iter_solve(self, num_iter = 2):
        pass
