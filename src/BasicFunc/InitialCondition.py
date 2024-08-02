#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:48:02 2024

@author: bojin
"""
from src.Deps import *


class SetInitialCondition:
    """
    
    """
    def __init__(self, flag, ic=None, fw=None, noise=False, timestamp=0.0, mesh=None, element_in = None, element_out = None):
        """
        

        Parameters
        ----------
        flag : TYPE
            DESCRIPTION.
        ic : TYPE, optional
            DESCRIPTION. The default is None.
        fw : TYPE, optional
            DESCRIPTION. The default is None.
        noise : TYPE, optional
            DESCRIPTION. The default is False.
        timestamp : TYPE, optional
            DESCRIPTION. The default is 0.0.
        mesh : TYPE, optional
            DESCRIPTION. The default is None.
        element_in : TYPE, optional
            DESCRIPTION. The default is None.
        element_out : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if noise is not False and element_out is not None:
            vec_noise=self.__setnoise(noise, element_out)
            
        if ic is not None and fw is not None:

            if flag == 0: # steady newton solver
                self.__setfunc(ic, fw, timestamp)
            if flag == 1: # transient newton solver
                self.__setfunc(ic, fw, timestamp)
                if vec_noise is not None:
                    fw.vector()[:] += np.ascontiguousarray(vec_noise)
            if flag == 2: # transient IPCS solver (MPI)
                self.__setfunc_parallel(ic, fw, mesh, timestamp, element_in = element_in, element_out = element_out)
                if vec_noise is not None:
                    fw[0].vector()[:] += np.ascontiguousarray(vec_noise[0])
                    fw[1].vector()[:] += np.ascontiguousarray(vec_noise[1])
    
    def __readfile(self, path, fw, timestamp):
        """
        

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        fw : TYPE
            DESCRIPTION.
        timestamp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        timeseries_flow = TimeSeries(path)
        timeseries_flow.retrieve(fw.vector(), timestamp)
        
    def __readfile_hdf5(self, hdf, fw, timestamp, label, mesh):
        """
        

        Parameters
        ----------
        hdf : TYPE
            DESCRIPTION.
        fw : TYPE
            DESCRIPTION.
        timestamp : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.
        mesh : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        time_bool = False
        for ts in range(hdf.attributes(label)['count']):
            time_bool = (hdf.attributes(label+'/vector_' + str(ts))['timestamp'] == timestamp)
            if time_bool:
                print('Read vector_'+str(ts)+'as the initial condition')
                hdf.read(fw, label+'/vector_' + str(ts))
                break
        if time_bool is False:
            info('Initial Timestamp not found !!!')

        
    def __setfunc(self, ic, fw, timestamp):
        """
        

        Parameters
        ----------
        ic : TYPE
            DESCRIPTION.
        fw : TYPE
            DESCRIPTION.
        timestamp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(ic) is type("path"):
            self.__readfile(ic, fw, timestamp)
        elif type(ic) is type(fw):
            assign(fw, ic)
        else:
            info("Wrong Format of Initial Conidtion (Please give a path or function)")
            
            
    def __setfunc_parallel(self, ic, fw, mesh, timestamp, element_in = None, element_out = None):
        """
        

        Parameters
        ----------
        ic : TYPE
            DESCRIPTION.
        fw : TYPE
            DESCRIPTION.
        mesh : TYPE
            DESCRIPTION.
        timestamp : TYPE
            DESCRIPTION.
        element_in : TYPE, optional
            DESCRIPTION. The default is None.
        element_out : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        # set parallel in Decoupled element
        if type(ic) is type("path"): # if a path is given
            hdf = HDF5File(mesh.mpi_comm(), path, 'r')
            if hdf.has_dataset('Velocity') and hdf.has_dataset('Pressure'): # stored in two datasets
                self.__readfile_hdf5(hdf, fw[0], timestamp, 'Velocity', mesh)
                self.__readfile_hdf5(hdf, fw[1], timestamp, 'Pressure', mesh)
            elif hdf.has_dataset('Coupled Field'):
                self.__readfile_hdf5(hdf, element_in.w, timestamp, 'Coupled Field', mesh)
                assign(fw[0], project(element_in.u, element_out.functionspace_V, solver_type='gmres'))
                assign(fw[1], project(element_in.p, element_out.functionspace_Q, solver_type='gmres'))
        else:
            self.__set_fun(ic[0], fw[0])
            self.__set_fun(ic[1], fw[1])

    def __setnoise(self, noise, element_out):
        """
        

        Parameters
        ----------
        noise : TYPE
            DESCRIPTION.
        element_out : TYPE
            DESCRIPTION.

        Returns
        -------
        vec_noise : TYPE
            DESCRIPTION.

        """
        vec_noise = None
        if element_out.type == 'TaylorHood':
            if noise is True or type(noise) is float:
                pertub = (2 * np.random.rand(element_out.functionspace.dim()) - 1) * float(noise)
                vec_noise = np.ascontiguousarray(pertub)
            elif type(noise) is np.ndarray:
                vec_noise = np.ascontiguousarray(noise)
            else:
                info("Wrong Format of Initial Noise (Please give a float or np.ndarray)")
                
        elif element_out.type == 'Decoupled':
            if noise is True or type(noise) is float: # if noise is np.random.rand 
                pertub_v = (2 * np.random.rand(element_out.functionspace_V.dim()) - 1) * float(noise)
                vec_noise = (np.ascontiguousarray(pertub_v),)
                pertub_q = (2 * np.random.rand(element_out.functionspace_Q.dim()) - 1) * float(noise)
                vec_noise += (np.ascontiguousarray(pertub_q),)
                
            elif type(noise[0]) is np.ndarray and type(noise[1]) is np.ndarray: # if noise is np.ndarray
                vec_noise = (np.ascontiguousarray(noise[0]),)
                vec_noise += (np.ascontiguousarray(noise[1]),)
            else:
                info("Wrong Format of Initial Noise (Please give a float or tuple of two np.ndarray)")
        else:
            info("Unknow element type in initial noise setting")
        return vec_noise
          

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


