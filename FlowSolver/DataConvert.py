from __future__ import print_function
from fenics import *
import numpy as np
from .FiniteElement import *

"""This module provides functions
"""

def DataConvert_mesh(mesh1,mesh2, path1, path2, etype = 'TaylorHood', time = 0.0, time_store = 0.0):
    """
    Data convert between different meshes: from mesh1 to mesh2
    :param mesh1:
    :param mesh2:
    :param path1:
    :param path2:
    :param etype:
    :return:
    """

    parameters["allow_extrapolation"] = True
    if etype == 'TaylorHood' and type(path1) is not dict and type(path2) is not dict:
        element_1 = TaylorHood(mesh=mesh1)
        w1 = element_1.w
        element_2 = TaylorHood(mesh=mesh2)
        timeseries_1 = TimeSeries(path1)
        timeseries_1.retrieve(w1.vector(), time)
        w2 = project(w1, element_2.functionspace, solver_type='gmres')
        timeseries_2 = TimeSeries(path2)
        timeseries_2.store(w2.vector(), time_store)
    if etype == 'SplitPV' and type(path1) is dict and type(path2) is dict:
        element_1 = SplitPV(mesh=mesh1)
        u1 = element_1.u
        p1 = element_1.p
        timeseries_v1 = TimeSeries(path1['Velocity'])
        timeseries_v1.retrieve(u1.vector(), time)
        timeseries_p1 = TimeSeries(path1['Pressure'])
        timeseries_p1.retrieve(p1.vector(), time)

        element_2 = SplitPV(mesh=mesh2)
        u2 = project(u1, element_2.functionspace_V, solver_type='gmres')
        p2 = project(p1, element_2.functionspace_Q, solver_type='gmres')
        timeseries_v2 = TimeSeries(path2['Velocity'])
        timeseries_v2.store(u2.vector(), time_store)
        timeseries_p2 = TimeSeries(path2['Pressure'])
        timeseries_p2.store(p2.vector(), time_store)

def DataConvert_element(mesh1,mesh2, path1, path2, etype = 'TaylorHood2SplitPV', time = 0.0, time_store = 0.0):
    parameters["allow_extrapolation"] = True
    if etype == 'TaylorHood2SplitPV' and type(path1) is not dict and type(path2) is dict:
        element_1 = TaylorHood(mesh=mesh1)
        element_2 = SplitPV(mesh=mesh2)
        timeseries_1 = TimeSeries(path1)
        timeseries_1.retrieve(element_1.w.vector(), time)
        u2 = project(element_1.u, element_2.functionspace_V, solver_type='gmres')
        p2 = project(element_1.p, element_2.functionspace_Q, solver_type='gmres')
        timeseries_v2 = TimeSeries(path2['Velocity'])
        timeseries_v2.store(u2.vector(), time_store)
        timeseries_p2 = TimeSeries(path2['Pressure'])
        timeseries_p2.store(p2.vector(), time_store)
    elif etype == 'SplitPV2TaylorHood' and type(path2) is not dict and type(path1) is dict:
        element_1 = SplitPV(mesh=mesh1)
        element_2 = TaylorHood(mesh=mesh2)
        timeseries_v1 = TimeSeries(path1['Velocity'])
        timeseries_v1.retrieve(element_1.u.vector(), time)
        timeseries_p1 = TimeSeries(path1['Pressure'])
        timeseries_p1.retrieve(element_1.p.vector(), time)
        u2 = project(element_1.u, element_2.functionspace.sub(0), solver_type='gmres')
        p2 = project(element_1.p, element_2.functionspace.sub(1), solver_type='gmres')
        element_2.u.vector()[:] = u2.vector()[:]
        element_2.p.vector()[:] = p2.vector()[:]
        timeseries_2 = TimeSeries(path2)
        timeseries_2.store(element_2.w.vector(), time_store)