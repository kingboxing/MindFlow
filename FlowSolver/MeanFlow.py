from __future__ import print_function
from fenics import *
import numpy as np
from .FiniteElement import *



class MeanFlow():
    def __init__(self, element=None, path=None, start=None, end=None, dt=None):
        if element.type == 'TaylorHood':
            self.element = TaylorHood(mesh=element.mesh,dim=element.dimension, order=element.order, constrained_domain=element.constrained_domain)
            self.path = path
        if element.type == 'Split' and type(path) is dict:
            self.element = SplitPV(mesh=element.mesh,dim=element.dimension, order=element.order, constrained_domain=element.constrained_domain)
            self.path_v = path['Velocity']
            self.path_p = path['Pressure']
        self.start = start
        self.end = end
        self.dt = dt
        self.meanflow()

    def meanflow(self):
        time_series = np.arange(self.start,self.end+self.dt,self.dt)
        if self.element.type == 'TaylorHood':
            w = self.element.w
            timeseries = TimeSeries(self.path)
            mean = self.element.add_functions()
            timeseries.retrieve(mean.vector(), self.start)
            timeseries.retrieve(w.vector(), self.end)
            mean.vector()[:] = (mean.vector()[:]+w.vector()[:])/2.0
            for t in time_series[1:-1]:
                print('time= ', t)
                timeseries.retrieve(w.vector(), t)
                mean.vector()[:] += w.vector()[:]
            mean.vector()[:] = mean.vector()[:]/(len(time_series)-1)
            self.mean = mean
        if self.element.type == 'Split':
            u = self.element.u
            p = self.element.p
            timeseries_v = TimeSeries(self.path_v)
            timeseries_p = TimeSeries(self.path_p)
            mean_v, mean_p = self.element.add_functions()
            timeseries_v.retrieve(mean_v.vector(), self.start)
            timeseries_v.retrieve(u.vector(), self.end)
            mean_v.vector()[:] = (mean_v.vector()[:] + u.vector()[:]) / 2.0
            timeseries_p.retrieve(mean_p.vector(), self.start)
            timeseries_p.retrieve(p.vector(), self.end)
            mean_p.vector()[:] = (mean_p.vector()[:] + p.vector()[:]) / 2.0
            for t in time_series[1:-1]:
                print('time= ', t)
                timeseries_v.retrieve(u.vector(), t)
                mean_v.vector()[:] += u.vector()[:]
                timeseries_p.retrieve(p.vector(), t)
                mean_p.vector()[:] += p.vector()[:]
            mean_v.vector()[:] = mean_v.vector()[:] / (len(time_series) - 1)
            mean_p.vector()[:] = mean_p.vector()[:] / (len(time_series) - 1)
            self.mean_v = mean_v
            self.mean_p = mean_p

    def dataconvert(self, element2, path2):
        """
        Data convert between different meshes
        :param element2:
        :param path2:
        :return:
        """
        parameters["allow_extrapolation"] = True
        if self.element.type == 'TaylorHood' and self.element.type == element2.type:
            w1 = self.mean
            w2 = project(w1, element2.functionspace, solver_type='gmres')
            timeseries_2 = TimeSeries(path2)
            timeseries_2.store(w2.vector(), 0.0)
        if self.element.type == 'Split' and self.element.type == element2.type and type(path2) is dict:
            u1 = self.mean_v
            p1 = self.mean_p
            u2 = project(u1, element2.functionspace, solver_type='gmres')
            p2 = project(p1, element2.functionspace, solver_type='gmres')
            timeseries_v = TimeSeries(path2['Velocity'])
            timeseries_v.store(u2.vector(), 0.0)
            timeseries_p = TimeSeries(path2['Pressure'])
            timeseries_p.store(p2.vector(), 0.0)