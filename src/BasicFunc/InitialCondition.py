#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:48:02 2024

@author: bojin
"""
from ..Deps import *


class SetInitialCondition:
    """
    Set initial conditions for FEniCS simulations, with optional noise and support for parallel execution.
    """
    def __init__(self, flag, ic=None, fw=None, noise=False, timestamp=0.0, mesh=None, element_in = None, element_out = None):
        """
        Initialize the initial condition setup.

        Parameters
        ----------
        flag : int
            Determines the type of solver (0: steady, 1: transient, 2: IPCS solver).
        ic : str or Function, optional
            Initial condition, either as a file path or a FEniCS function. Default is None.
        fw : Function or list of Functions, optional
            The function(s) to which the initial condition is applied. Default is None.
        noise : bool, float, or ndarray, optional
            Noise to be added to the initial condition. Default is False (no noise).
        timestamp : float, optional
            Timestamp for retrieving the initial condition from a time series. Default is 0.0.
        mesh : Mesh, optional
            The mesh associated with the simulation. Required for parallel solvers. Default is None.
        element_in : object, optional
            The input element object (e.g., TaylorHood). Default is None.
        element_out : object, optional
            The output element object (e.g., TaylorHood). Default is None.
        """
        vec_noise = self._set_noise(noise, element_out) if noise and element_out else None
        
        if ic and fw:
            if flag in [0, 1]:# steady newton solver/transient newton solver
                self._set_function(ic, fw, timestamp)
                if flag == 1 and vec_noise:
                    fw.vector()[:] += vec_noise
            elif flag == 2: # transient IPCS solver (MPI)
                self._set_function_parallel(ic, fw, mesh, timestamp, element_in = element_in, element_out = element_out)
                if vec_noise:
                    fw[0].vector()[:] += vec_noise[0]
                    fw[1].vector()[:] += vec_noise[1]
    
    def _read_timeseries(self, path, fw, timestamp):
        """
        Read a time series file and apply it to the function.

        Parameters
        ----------
        path : str
            Path to the time series file.
        fw : Function
            The function to apply the time series to.
        timestamp : float
            The timestamp to retrieve from the time series.
        """
        
        timeseries_flow = TimeSeries(path)
        timeseries_flow.retrieve(fw.vector(), timestamp)
#%%        
    def _read_hdf5(self, hdf, fw, timestamp, label, mesh):
        """
        Read an HDF5 file and apply it to the function.

        Parameters
        ----------
        hdf : HDF5File
            The HDF5 file to read from.
        fw : Function
            The function to apply the HDF5 data to.
        timestamp : float
            The timestamp to retrieve from the HDF5 file.
        label : str
            The label of the dataset in the HDF5 file.
        mesh : Mesh
            The mesh associated with the HDF5 file.
        """
        for ts in range(hdf.attributes(label)['count']):
            if hdf.attributes(label + '/vector_' + str(ts))['timestamp'] == timestamp:
                print(f'Read vector_{ts} as the initial condition')
                hdf.read(fw, label + '/vector_' + str(ts))
                return
        info('Initial Timestamp not found!')
        
#%%
        
    def _set_function(self, ic, fw, timestamp=0.0):
        """
        Set the initial condition for a function.

        Parameters
        ----------
        ic : str or Function
            Initial condition, either as a file path or a FEniCS function.
        fw : Function
            The function to apply the initial condition to.
        timestamp : float, optional
            Timestamp for retrieving the initial condition from a time series. Default is 0.0.
        """
        if isinstance(ic, str):
            self._read_timeseries(ic, fw, timestamp)
        elif isinstance(ic, function.function.Function) and isinstance(fw, function.function.Function):
            assign(fw, ic)
        else:
            info("Invalid initial condition format (expected path or function).")
           
#%%
    def _set_function_parallel(self, ic, fw, mesh, timestamp, element_in = None, element_out = None):
        """
        Set the initial condition for a parallel IPCS solver.

        Parameters
        ----------
        ic : str or Function
            Initial condition, either as a file path or a FEniCS function.
        fw : list of Functions
            The functions to apply the initial condition to.
        mesh : Mesh
            The mesh associated with the simulation.
        timestamp : float
            Timestamp for retrieving the initial condition. Default is 0.0.
        element_in : object, optional
            The input element object (e.g., TaylorHood). Default is None.
        element_out : object, optional
            The output element object (e.g., TaylorHood). Default is None.
        """
        
        # set parallel in Decoupled element
        if isinstance(ic, str): # if a path is given
            hdf = HDF5File(mesh.mpi_comm(), ic+'.h5', 'r')
            if hdf.has_dataset('Velocity') and hdf.has_dataset('Pressure'): # stored in two datasets
                self._read_hdf5(hdf, fw[0], timestamp, 'Velocity', mesh)
                self._read_hdf5(hdf, fw[1], timestamp, 'Pressure', mesh)
            else:
                if hdf.has_dataset('Coupled Field'):
                    hdf.rename(hdf.label(), 'Coupled Field')
                    self._read_hdf5(hdf, element_in.w, timestamp, 'Coupled Field', mesh)
                elif element_in.type == 'TaylorHood':
                    self._read_timeseries(ic, element_in.w, timestamp)  
                assign(fw[0], element_in.w.sub(0))
                assign(fw[1], element_in.w.sub(1))
                
        elif element_in.type == element_out.type: # type is 'Decoupled', ic is a tuple
            self._set_function(ic[0], fw[0])
            self._set_function(ic[1], fw[1])
        elif element_in.type == 'TaylorHood': # ic is a Function
            self._set_function(ic.sub(0), fw[0])
            self._set_function(ic.sub(1), fw[1])
#%%
    def _set_noise(self, noise, element_out):
        """
        Generate noise to add to the initial condition.

        Parameters
        ----------
        noise : bool, float, or ndarray
            The noise level or array to add.
        element_out : object
            The output element object (e.g., TaylorHood).

        Returns
        -------
        vec_noise : ndarray or tuple of ndarrays
            The noise vector(s).
        """
        
        vec_noise = None
        if element_out.type == 'TaylorHood':
            if noise is True or isinstance(noise, (float, np.floating)):
                pertub = (2 * np.random.rand(element_out.functionspace.dim()) - 1) * float(noise)
                vec_noise = np.ascontiguousarray(pertub)
            elif isinstance(noise, np.ndarray):
                vec_noise = np.ascontiguousarray(noise)
            else:
                info("Invalid noise format (expected float or ndarray).")
                
        elif element_out.type == 'Decoupled':
            if noise is True or isinstance(noise, (float, np.floating)): # if noise is np.random.rand 
                pertub_v = (2 * np.random.rand(element_out.functionspace_V.dim()) - 1) * float(noise)
                vec_noise = (np.ascontiguousarray(pertub_v),)
                pertub_q = (2 * np.random.rand(element_out.functionspace_Q.dim()) - 1) * float(noise)
                vec_noise += (np.ascontiguousarray(pertub_q),)
            elif all(isinstance(n, np.ndarray) for n in noise): # if noise is np.ndarray
                vec_noise = (np.ascontiguousarray(noise[0]),)
                vec_noise += (np.ascontiguousarray(noise[1]),)
            else:
                info("Invalid noise format (expected float or tuple of ndarrays).")
        else:
            info("Unknown element type in initial noise setting")
        return vec_noise
    
    def visit_hdf5(self, path):
        """
        Visit and list all attributes in an HDF5 file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        """
        
        import h5py
        with h5py.File(path+'.h5', 'r') as h5file:
            # List all the datasets in the file    
            h5file.visit(print)
          

## abandoned

"""

def InitialCondition(flag=None, ic=None, noise=False, timestamp=0.0, mesh = None, element = None, fu = None, fp = None, fw = None):
    match flag:
        case 0 :# steady newton solver
            if ic is not None:
                assign(self.w, ic)
        
        case 1 :# transient newton solver
            if ic is not None:
                if type(ic) is type("path"):
                    timeseries_flow = TimeSeries(ic)
                    timeseries_flow.retrieve(self.eqn.fw[1].vector(), timestamp)
                elif type(ic) is type(self.eqn.fw[1]):
                    assign(self.eqn.fw[1], ic)
                else:
                    info("Wrong Format of Initial Conidtion (Please give a path or function)")
            
            if noise is True or type(noise) is float:
                pertub = (2 * np.random.rand(self.element.functionspace.dim()) - 1) * float(noise)
                self.eqn.fw[1].vector()[:] += np.ascontiguousarray(pertub)
            elif type(noise) is np.ndarray:
                self.eqn.fw[1].vector()[:] += np.ascontiguousarray(noise)
            else:
                info("Wrong Format of Initial Noise (Please give a float or np.ndarray)")
            
        case 2:
            if ic is not None: # setup initial condition 
                if type(ic) is type("path"): # if a path is given
                    time_bool = False
                    hdf = HDF5File(self.mesh.mpi_comm(), ic, 'r')
                    if hdf.has_dataset('Velocity') and hdf.has_dataset('Pressure'): # stored in two datasets
                        for ts in range(hdf.attributes('Velocity')['count']):
                            time_bool = (hdf.attributes('Velocity/vector_' + str(ts))['timestamp'] == timestamp)
                            if time_bool:
                                print('Set vector_'+str(ts)+'as the initial condition')
                                hdf.read(self.eqn.fu[1], 'Velocity/vector_' + str(ts))
                                hdf.read(self.eqn.fp[1], 'Pressure/vector_' + str(ts))
                                break
                    elif hdf.has_dataset('Coupled Field'):
                        for ts in range(hdf.attributes('Coupled Field')['count']):
                            time_bool = (hdf.attributes('Coupled Field/vector_' + str(ts))['timestamp'] == timestamp)
                            if time_bool:
                                print('Set vector_' + str(ts) + ' as the initial condition')
                                element_base = TaylorHood(mesh=self.mesh)
                                element_s = Decoupled(mesh=self.mesh)
                                hdf.read(element_base.w, 'Coupled Field/vector_' + str(ts))
                                (vel, pre) = split(element_base.w)
                                self.fu[1] = project(vel, element_s.functionspace_V, solver_type='gmres')
                                self.fp[1] = project(pre, element_s.functionspace_Q, solver_type='gmres')
                                break
                    if time_bool is False:
                        raise Exception('Timestamp not found')   
                elif type(ic[0]) is type(self.eqn.fu[1]) and type(ic[1]) is type(self.eqn.fp[1]): 
                    # if a tuple of functions is given
                    assign(self.eqn.fu[1], ic[0])
                    assign(self.eqn.fp[1], ic[1])
                else:
                    info("Wrong Format of Initial Conidtion (Please give a path or function)")
            
            # setup initial noise
            if noise is True or type(noise) is float: # if noise is np.random.rand 
                pertub_v = (2 * np.random.rand(self.element.functionspace_V.dim()) - 1) * float(noise)
                self.eqn.fu[1].vector()[:] += np.ascontiguousarray(pertub_v)
                
                pertub_q = (2 * np.random.rand(self.element.functionspace_Q.dim()) - 1) * float(noise)
                self.eqn.fp[1].vector()[:] += np.ascontiguousarray(pertub_q)
                
            elif type(noise[0]) is np.ndarray and type(noise[1]) is np.ndarray: # if noise is np.ndarray
                self.eqn.fu[1].vector()[:] += np.ascontiguousarray(noise[0])
                self.eqn.fp[1].vector()[:] += np.ascontiguousarray(noise[1])
            else:
                info("Wrong Format of Initial Noise (Please give a float or tuple of two np.ndarray)")
"""


