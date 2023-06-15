from __future__ import print_function
from fenics import *
import numpy as np
import gc
from scipy.sparse import csr_matrix, csc_matrix,isspmatrix_csc
import scipy.sparse.linalg as spla
from .MatrixAssemble import MatrixAssemble
from .EigenAnalysis import EigenAnalysis
import os,sys,inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir)
from SVD.MatrixOperator import MatInv
from Boundary.Set_BoundaryCondition import BoundaryCondition
from FlowSolver.FiniteElement import TaylorHood
from FlowSolver.Tools import *


"""This module provides the class that calculates frequency response of a flow field
"""

class FrequencyAnalysis(MatrixAssemble):
    def __init__(self, mesh, boundary, omega=None, sigma=None,s=None, nu=None, path=None, baseflow=None,order=(2,1)):
        self.element_order=order
        element = TaylorHood(mesh=mesh,order=self.element_order) # initialise finite element space
        # inherit attributes : functionspace, boundries, bcs; methods: assemblematrix, assemblevector
        #BoundaryCondition.__init__(self, Functionspace=element.functionspace, boundary=boundary)
        self.BC = BoundaryCondition(Functionspace=element.functionspace, boundary=boundary)
        self.functionspace = element.functionspace
        self.boundaries = boundary.get_domain() # Get the FacetFunction on given mesh
        MatrixAssemble.__init__(self)
        # create public attributes
        self.mesh = mesh
        self.path = path
        self.baseflow=baseflow
        if s is not None:
            if omega is None and sigma is None:
                self.omega=Constant(np.imag(s))
                self.sigma=Constant(np.real(s))
            else:
                raise ValueError('Please specify consistent Frequency and Growth Rate values')  
        if s is None:
            self.omega = omega
            self.sigma = sigma
            if omega is None:
                self.omega = Constant(0.0)
            if sigma is None:
                self.sigma = Constant(0.0)
        self.nu = nu
        self.n = FacetNormal(self.mesh)
        self.ds = boundary.get_measure()
        # assign functions for the expression convenience
        self.u=element.tu
        self.p=element.tp
        self.v=element.v
        self.q=element.q
        self.w=element.w
        self.u_base=element.u
        self.p_base=element.p
        # get the coordinates of the whole functionspace
        self.Coor_fun = self.functionspace.tabulate_dof_coordinates().reshape((-1, 2))
        # initialize input/output
        self.IO_vec=InputOutput(mesh=self.mesh, boundary=boundary, omega=self.omega, sigma=self.sigma, nu=self.nu, element=element)
        # prepare
        # if path is not None:
        #     self.__set_baseflow()
        # if nu is not None:
        #     self.__NS_expression()

    def set_boundarycondition(self, boucon, mark):
        try:
            test = boucon['FunctionSpace']
        except:
            boucon['FunctionSpace']=None
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
        else:
            pass
        
        try:
            test = boucon['Value']
        except:
            boucon['Value']=None
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
        else:
            pass
        
        if (boucon['FunctionSpace'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet'] or boucon['Value'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet']):
            self.freeoutlet=True
        else:
            self.BC.set_boundarycondition(boucon,mark)


    def __to_int64(self, x):
        y = csc_matrix(x).copy()
        y.indptr = y.indptr.astype(np.int64)
        y.indices = y.indices.astype(np.int64)
        return y

    def __NablaGradU(self,mesh,baseflow):
        V = TensorFunctionSpace(mesh,"Lagrange",1)
        (baseu,basep)=baseflow.split()
        gradu=project(nabla_grad(baseu),V, solver_type='mumps')
        return gradu    
    
    def __set_baseflow(self):
        """Assign base flow to the function w
        """
        if self.path is not None and self.baseflow is None:
            timeseries_flow = TimeSeries(self.path)
            timeseries_flow.retrieve(self.w.vector(), 0.0)
        if self.baseflow is not None and self.path is None:
            try:
                assign(self.w,self.baseflow)
            except:
                assign(self.w.sub(0),self.baseflow)
            else:
                pass
#        timeseries_flow = TimeSeries(self.path)
#        timeseries_flow.retrieve(self.w.vector(), 0.0)
                
    # test functionspace with order=(1,1)
    def set_baseflow_test(self): 
        
        timeseries_flow = TimeSeries(self.path)
        element = TaylorHood(mesh=self.mesh)
        timeseries_flow.retrieve(element.w.vector(), 0.0)
        
        baseflow = interpolate(element.w, self.functionspace)
        assign(self.w,baseflow)
        self.nablagradu=self.__NablaGradU(self.mesh,element.w)

    def NS_expression_test(self):
        self.ai = self.omega * inner(self.u, self.v) * dx # imaginary part
        self.ar = self.sigma * inner(self.u, self.v) * dx + (inner(dot(self.u_base, nabla_grad(self.u)), self.v) +
                   inner(dot(self.u, self.nablagradu), self.v) -
                   div(self.v) * self.p + self.nu * inner(grad(self.u), grad(self.v))
                   + self.q * div(self.u)) * dx


    def __NS_expression(self):
        """Create the weak form of the N-S equations in the frequency domain
        wj*u+U*grad(u)+u*grad(U)=-grad(p)+nu*div(grad(u))
        div(u)=0

        """
        self.ai = self.omega * inner(self.u, self.v) * dx # imaginary part
        self.ar = self.sigma * inner(self.u, self.v) * dx + (inner(dot(self.u_base, nabla_grad(self.u)), self.v) +
                   inner(dot(self.u, nabla_grad(self.u_base)), self.v) -
                   div(self.v) * self.p + self.nu * inner(grad(self.u), grad(self.v))
                   + self.q * div(self.u)) * dx
        # do not need these if Dirichlet or zero Neumann boundary conditions applied on all boundaries         
        try:
            self.freeoutlet
        except:
            self.ar = self.ar+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
        else:
            if self.freeoutlet is not True:
                self.ar = self.ar+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds


    def __assemble_NSfunction(self,MatK=None):
        """Assemble the matrix L that represents N-S equations as L*[u, p]^T = f

        """
        Ar_sparray = self.assemblematrix(self.ar,self.BC.bcs) # assemble the real part
        if MatK is not None:
            Ar_sparray=Ar_sparray+MatK
        Ai_sparray = self.assemblematrix(self.ai,self.BC.bcs) # assemble the imag part
        # matrix has only ones in diagonal for rows specified by the boundary condition
        I_sparray = self.assemblematrix(Constant(0.0)*self.ai,self.BC.bcs)
        # Complex matrix L with boundary conditions
        L = Ar_sparray + Ai_sparray.multiply(1j)-I_sparray.multiply(1j)
        # Change the format to CSC, it will be more efficient while LU decomposition
        self.L = L.tocsc()

    def __solveroptions(self,useUmfpack,solver_type='direct'):
        L=self.L
        if solver_type=='direct':
            info('LU decomposition...')
            gc.collect()
            if not isspmatrix_csc(L):
                L=L.tocsc()
            if useUmfpack is True:
                self.Linv=MatInv(L,lusolver='umfpack')
            else: 
                try:
                    self.Linv=MatInv(L,lusolver='mumps')
                except:
                    info('switch to SuperLU solver')
                    try:
                        self.Linv=MatInv(L,lusolver='superlu',echo=True)
                    except:
                        info('switch to useUmfpack solver')
                        self.Linv=MatInv(L,lusolver='umfpack')
                        info('Using Umfpack solver')
                    else:
                        info('Using SuperLU solver')
                else:
                    info('Using Mumps solver')
            info('Done.')
        elif solver_type=='iterative':
            self.Minv=precondition_jacobi(L, useUmfpack=useUmfpack)
            self.L=L

            

    def __Input(self, options):
        if 'Vector' in options:
            self.input_vec=np.matrix(options['Vector'])
        else:
            self.input_vec = eval("self.IO_vec."+self.IO_vec.dict_inputs[options['Variable']])#+"[0]")
            
        if self.input_vec.shape[1]==np.max(self.input_vec.shape) and self.input_vec.shape[0]!=self.input_vec.shape[1]:
            self.input_vec = self.input_vec.transpose() ## ensure the shape of input vector as (n, k) with k is the number of inputs

    def __Output(self, options):
        if 'Vector' in options:
            self.output_vec ==np.matrix(options['Vector'])
        else:
            self.output_vec = eval("self.IO_vec."+self.IO_vec.dict_outputs[options['Variable']])#+"[0]")
            
        if self.output_vec.shape[0]==np.max(self.output_vec.shape) and self.output_vec.shape[0]!=self.output_vec.shape[1]:
            self.output_vec = self.output_vec.transpose() ## ensure the shape of output vector as (k, n) with k is the number of outputs


    def __Problem(self,solver_type='direct',v0=None):
        if solver_type=='direct' and self.Linv.lusolver != 'umfpack':
            self.state = self.Linv.matvec(self.input_vec)
        elif solver_type=='direct' and self.Linv.lusolver == 'umfpack':#umfpack solver only accept 1d array?
            self.state = np.matrix(np.zeros(self.input_vec.shape),dtype='complex')
            for i in range(self.input_vec.shape[1]):
                self.state[:,i]=np.matrix(self.Linv.matvec(np.reshape(np.asarray(self.input_vec[i,:]),len(self.input_vec[i,:].T)))).T
            
        elif solver_type=='iterative':
            self.state = np.matrix(spla.gmres(self.L, self.input_vec,x0=v0, M=self.Minv)[0]).T
        
        gain = self.output_vec*self.state
        return gain

    def solve(self, omega=None, sigma=None,s=None, nu=None, path=None,baseflow=None, useUmfpack=False,options={}, ReuseLU=False,solver_type='direct',v0=None,MatK=None):
        if ReuseLU is False:
            if self.nu is None and nu is None:
                raise ValueError('Please specify Kinematic viscosity')
            if self.path is None and path is None and self.baseflow is None and baseflow is None:
                raise ValueError('Please specify the path of the base flow')
            if path is not None and baseflow is None:
                self.path=path
                self.baseflow=None
            if baseflow is not None and path is None:
                self.baseflow=baseflow
                self.path=None
            if nu is not None:
                self.nu = nu
                self.IO_vec.nu = nu    
                
            if s is None:
                if omega is not None:
                    self.omega = omega
                    self.IO_vec.omega = omega
                if sigma is not None:
                    self.sigma = sigma
                    self.IO_vec.sigma = sigma
            if s is not None:
                if omega is not None or sigma is not None:
                    raise ValueError('Please specify consistent Frequency and Growth Rate values')                              
                if omega is None and sigma is None:
                    self.omega=Constant(np.imag(s))
                    self.sigma=Constant(np.real(s))
            
            self.__set_baseflow()
            self.__NS_expression()
            self.__assemble_NSfunction(MatK=MatK)
            self.__solveroptions(useUmfpack=useUmfpack,solver_type=solver_type)
            
            self.IO_vec.bcs = self.BC.bcs

        """options={'Input':  {'Variable': 'Acceleration','Direction': 1},
                    'Output': {'Variable': 'BodyForce',   'Direction': 1, 'BodyMark': 5}}
        """
        if 'Input' in options:
            self.__Input(options['Input'])
        else:
            raise ValueError('Please specify input vector')
        
        if 'Output' in options:
            self.__Output(options['Output'])
        else:
            self.output_vec = np.matrix(np.zeros((1,np.max(self.input_vec.shape))))
            info('Zero output')
        
        self.gain = self.__Problem(solver_type=solver_type,v0=v0)
        info('Re=%g S=%g+%g i \t Gain_1st=%g+%g i' %(1.0/self.nu.values()[0], self.sigma.values()[0], self.omega.values()[0], np.real(self.gain[0,0]), np.imag(self.gain[0,0])))




class InputOutput(MatrixAssemble):
    def __init__(self,mesh=None, bcs=[],boundary=None, omega=None, sigma=None, nu=None, element=None):
        MatrixAssemble.__init__(self)
        self.bcs = bcs
        if sigma is None:
            sigma=Constant(0.0)
        if omega is None:
            omega=Constant(0.0)
        self.omega = omega
        self.sigma = sigma
        self.element = element
        self.mesh = mesh
        self.nu = nu
        self.n = FacetNormal(self.mesh)
        self.ds = boundary.get_measure()
        self.boundaries=boundary.get_domain()
        self.__dict_methods()

    def __dict_methods(self):
        self.dict_inputs = {'OscillateDisplacement'   : "OscillateDisplacement(direction=options['Direction'])",
                            'OscillateVelocity'       : "OscillateVelocity(direction=options['Direction'])",
                            'OscillateAcceleration'   : "OscillateAcceleration(direction=options['Direction'])",

                            'RotateAngle'       : "RotateAngle(bodymark=options['BodyMark'])",
                            'RotateVelocity'    : "RotateVelocity(bodymark=options['BodyMark'])",
                            'RotateAccleration' : "RotateAccleration(bodymark=options['BodyMark'])",

                            'GaussianForce_dual': "GaussianForce_dual(options['Radius'], options['Angle'], options['Sigma'])",
                            'GaussianForce_single': "GaussianForce_single(options['Radius'], options['Angle'], options['Sigma'])",
                            'GaussianForce': "GaussianForce(options['Radius'], options['Angle'], options['Sigma'],direction=options['Direction'])",

                            'PointForce'        : "PointForce(options['Radius'], options['Angle'], direction=options['Direction'])",
                            
                            
                            }

        self.dict_outputs = {'BodyForce'            : "BodyForce(bodymark=options['BodyMark'], direction=options['Direction'])",
                             'BodyForce_pressure'   : "BodyForce_pressure(bodymark=options['BodyMark'], direction=options['Direction'])",
                             'BodyForce_stress'     : "BodyForce_stress(bodymark=options['BodyMark'], direction=options['Direction'])",

                             'PointVelocity'        : "PointVelocity(coordinate=options['Coordinate'], direction=options['Direction'])",
                             'PointPressure'        : "PointPressure(coordinate=options['Coordinate'])",
                             'GaussianVelocity'     : "GaussianVelocity(coordinate=options['Coordinate'], direction=options['Direction'], sig=options['Sigma'])",
                             
                             'GaussianForce'         : "GaussianForce(options['Radius'], options['Angle'], options['Sigma'],direction=options['Direction'])"
                             }

    """Here for input vector
    """
    def OscillateDisplacement(self, direction=1):
        """
        d=Ae^[(sigma+omega*1i)*t]
        a=(sigma+omega*1i)^2*A*e^[(sigma+omega*1i)*t]
        """
        v = self.element.v

        input_exp_r = (self.sigma * self.sigma - self.omega*self.omega) * v[direction] * dx
        input_exp_i = Constant(2.0) * self.sigma * self.omega * v[direction] * dx

        input_vec_r = self.assemblevector(input_exp_r, self.bcs) - self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)
        input_vec_i = self.assemblevector(input_exp_i, self.bcs) - self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)
        input_vec = input_vec_r + input_vec_i*1j +\
                    (self.sigma.values()[0] + self.omega.values()[0] * 1j)*self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)

        return input_vec

    def OscillateVelocity(self, direction=1):
        """
        v=Ae^[(sigma+omega*1i)*t]
        a=(sigma+omega*1i)*A*e^[(sigma+omega*1i)*t]
        """
        v = self.element.v

        input_exp_r = Constant(1.0) * self.sigma * v[direction] * dx
        input_exp_i = Constant(1.0) * self.omega * v[direction] * dx

        input_vec_r = self.assemblevector(input_exp_r, self.bcs) - self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)
        input_vec_i = self.assemblevector(input_exp_i, self.bcs) - self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)

        input_vec = input_vec_r + input_vec_i*1j + self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)
        return input_vec

    def OscillateAcceleration(self,direction=1):
        """
        a=Ae^[(sigma+omega*1i)*t]
		
		BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(1.0),       'Boundary': 'top'},
							'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(1.0),       'Boundary': 'bottom'},
							'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,1.0)), 'Boundary': 'inlet'},
							'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'}
							}
        """
        v = self.element.v
        input_exp_r = Constant(1.0) * v[direction] * dx

        input_vec_r = self.assemblevector(input_exp_r, self.bcs) - self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)

        input_vec = input_vec_r + 1.0/(self.sigma.values()[0] + self.omega.values()[0] * 1j) * \
                                  self.assemblevector(Constant(0.0)*v[direction]*dx, self.bcs)
        return input_vec

    def __gaussianforce(self, radius, angle, sig, scale=1):
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), 2)
        center_x = radius*cos(angle/180.0*pi)
        center_y = radius*sin(angle/180.0*pi)
        sigma2 = sig**2
        force = Expression(
                'scale*pow(2.0*sigma2*pi,-1)*exp(-(pow(x[0]-center_x,2)+pow(x[1]-center_y,2))*pow(2.0*sigma2,-1))',
                degree=2, scale=scale,sigma2=sigma2, center_x=center_x, center_y=center_y)
        return force
        
    def GaussianForce(self, radius, angle, sig,scale=1,direction=0):
        """
        Gaussian force
        """
        v = self.element.v
        force = self.__gaussianforce(radius, angle, sig,scale=scale)
        
        directs=list(np.atleast_1d(direction))
        for d in directs:
            if directs.index(d)==0:
                input_exp = force*v[d]*dx
                input_vec = self.assemblevector(input_exp, self.bcs)
            elif directs.index(d)>0:
                input_exp = force*v[d]*dx
                input_vec=np.block([[input_vec],[self.assemblevector(input_exp, self.bcs)]])
            else:
                raise ValueError('Wrong force direction list')
        return input_vec#,[force]

    def GaussianForce_dual(self, radius, angle, sig, scale=1):
        """
        blowing and suction likewise
        """
        v = self.element.v
        force_upper = self.__gaussianforce(radius, angle, sig,scale=scale)
        force_lower = self.__gaussianforce(radius, -angle, sig,scale=scale)
        x_component = Constant(cos(angle/180.0*pi))
        y_component = Constant(sin(angle/180.0*pi))
        input_exp = (x_component*(force_upper-force_lower)*v[0]+y_component*(force_upper+force_lower)*v[1])*dx
        input_vec = self.assemblevector(input_exp, self.bcs)
        return input_vec#,[x_component*(force_upper-force_lower), y_component*(force_upper+force_lower)]

    def GaussianForce_single(self, radius, angle, sig, scale=1):
        """
        blowing and suction likewise
        """
        v = self.element.v
        force_upper = self.__gaussianforce(radius, angle, sig,scale=scale)
        x_component = Constant(cos(angle/180.0*pi))
        y_component = Constant(sin(angle/180.0*pi))
        input_exp = (x_component*(force_upper)*v[0]+y_component*(force_upper)*v[1])*dx
        input_vec = self.assemblevector(input_exp, self.bcs)
        return input_vec#,[x_component*(force_upper-force_lower), y_component*(force_upper+force_lower)]


    def PointForce(self,radius, angle, scale=1, direction=1,echo=True):
        """
        point force
        """
        center_x = radius*cos(angle/180.0*pi)
        center_y = radius*sin(angle/180.0*pi)
        coordinate=[center_x, center_y]
        
        input_vec=scale*self.PointVelocity(coordinate=coordinate, direction=direction,echo=echo)
        return input_vec

    def RotateAngle(self, bodymark=5):
        """
        alpha=Ae^[(sigma+omega*1i)*t]
        vel=(sigma+omega*1i)*A*e^[(sigma+omega*1i)*t]
        """
        v = self.element.v
        vel_rotate = Expression(('x[1]', '-x[0]'), degree=2)
        BC_rotate = DirichletBC(self.element.functionspace.sub(0), vel_rotate, self.boundaries, bodymark,method="geometric")
        input_exp = inner(Constant((0.0, 0.0)),v) * dx
        bcs = self.bcs+[BC_rotate]
        input_vec = (self.sigma.values()[0] + self.omega.values()[0]*1j)*self.assemblevector(input_exp, bcs)
        return input_vec

    def RotateVelocity(self, bodymark=5):
        """
        v=Ae^[(sigma+omega*1i)*t]
        """
        v = self.element.v
        vel_rotate = Expression(('x[1]', '-x[0]'), degree=2)
        BC_rotate = DirichletBC(self.element.functionspace.sub(0), vel_rotate, self.boundaries, bodymark,method="geometric")
        input_exp = inner(Constant((0.0, 0.0)),v) * dx
        bcs = self.bcs + [BC_rotate]
        input_vec = self.assemblevector(input_exp, bcs)
        return input_vec

    def RotateAccleration(self, bodymark=5):
        """
        a=Ae^[(sigma+omega*1i)*t]
        v=1/(sigma+omega*1i)*A*e^[(sigma+omega*1i)*t]
        """
        v = self.element.v
        vel_rotate = Expression(('x[1]', '-x[0]'), degree=2)
        BC_rotate = DirichletBC(self.element.functionspace.sub(0), vel_rotate, self.boundaries, bodymark,method="geometric")
        input_exp = inner(Constant((0.0, 0.0)),v) * dx
        bcs = self.bcs + [BC_rotate]
        input_vec = 1.0/(self.sigma.values()[0] + self.omega.values()[0] * 1j) * self.assemblevector(input_exp, bcs)
        return input_vec



    """Here for output
    """
    def BodyForce(self, bodymark, direction=1):
        u=self.element.tu
        p=self.element.tp
        I = Identity(u.geometric_dimension())
        D = sym(grad(u))
        force = (-p * I + 2.0 * self.nu * D)*self.n
        output_exp = force[direction] * self.ds(bodymark)
        output_vec = self.assemblevector(output_exp)
        return output_vec

    def BodyForce_pressure(self, bodymark, direction=1):
        u=self.element.tu
        p=self.element.tp
        I = Identity(u.geometric_dimension())
        force = (-p * I )*self.n
        output_exp = force[direction] * self.ds(bodymark)
        output_vec = self.assemblevector(output_exp)
        return output_vec

    def BodyForce_stress(self, bodymark, direction=1):
        u=self.element.tu
        D = sym(grad(u))
        force = (2.0 * self.nu * D)*self.n
        output_exp = force[direction] * self.ds(bodymark)
        output_vec = self.assemblevector(output_exp)
        return output_vec

    def __pointvelocity(self, coordinate=None, direction=1,echo=True):
        V=self.element.functionspace
        dofs_coor = V.tabulate_dof_coordinates().reshape((-1, 2))
        dofs_sub = V.sub(0).sub(direction).dofmap().dofs()
        subcoor = dofs_coor[dofs_sub, :]
        distance = (subcoor[:, 0]- coordinate[0])**2+(subcoor[:, 1]- coordinate[1])**2
        # index of vertical velocity closest to the coordiante
        ind_point = dofs_sub[np.argmin(distance)]
        output_vec = np.zeros((1, V.dim()))
        output_vec[0, ind_point] = 1.0
        if echo is True:
            info('The point closest to the specified coordinate is Coord = [%1.4f %1.4f], Polar = [%1.4f %1.4f]'
                 %(dofs_coor[ind_point,0], dofs_coor[ind_point,1], np.abs(dofs_coor[ind_point,0]+dofs_coor[ind_point,1]*1j), np.angle(dofs_coor[ind_point,0]+dofs_coor[ind_point,1]*1j)/pi*180))
        return np.matrix(output_vec)
        
    def PointVelocity(self, coordinate=None, direction=1,echo=True):
        directs=list(np.atleast_1d(direction))
        for d in range(np.size(directs)):
            if d==0:
                output_vec=self.__pointvelocity(coordinate=coordinate, direction=directs[d],echo=echo)
            elif d>0:
                output_vec=np.block([[output_vec],[self.__pointvelocity(coordinate=coordinate, direction=directs[d],echo=echo)]])
            else:
                raise ValueError('Wrong velocity direction list')
        return output_vec

    def PointPressure(self, coordinate=None):
        V=self.element.functionspace
        dofs_coor = V.tabulate_dof_coordinates().reshape((-1, 2))
        dofs_sub = V.sub(1).dofmap().dofs()
        subcoor = dofs_coor[dofs_sub, :]
        distance = (subcoor[:, 0]- coordinate[0])**2+(subcoor[:, 1]- coordinate[1])**2
        # index of vertical velocity closest to the coordiante
        ind_point = dofs_sub[np.argmin(distance)]
        output_vec = np.zeros((1, V.dim()))
        output_vec[0, ind_point] = 1.0
        info('The point closest to the specified coordinate is [%1.4f %1.4f]'
             %(dofs_coor[ind_point,0], dofs_coor[ind_point,1]))
        return np.matrix(output_vec)

    def __gaussianvelocity(self, coordinate=None, direction=1,sig=None, echo=True):
        v = self.element.v
        center_x = coordinate[0]
        center_y = coordinate[1]
        sigma2 = sig**2
        scale=1
        force = Expression(
                'scale*pow(2.0*sigma2*pi,-1)*exp(-(pow(x[0]-center_x,2)+pow(x[1]-center_y,2))*pow(2.0*sigma2,-1))',
                degree=self.element.order[0], scale=scale,sigma2=sigma2, center_x=center_x, center_y=center_y)

        output_exp = force*v[direction]*dx
        output_vec = self.assemblevector(output_exp)
        
        inds=self.element.w.sub(0).sub(direction).function_space().dofmap().dofs()
        Coor_fun = self.element.w.function_space().tabulate_dof_coordinates().reshape((-1, 2))[inds,:]
        rads = np.abs((Coor_fun[:,0]+Coor_fun[:,1]*1j)-(center_x+center_y*1j))
        output_vec[0,inds[rads>4*sig]]=0
        
        if echo is True:
            info('The integration of Gaussian distribution in direction '+str(direction)+' = %1.8f'
                 %(np.sum(output_vec)))
        return np.matrix(output_vec)

    def GaussianVelocity(self, coordinate=None, direction=1, sig=None, echo=True):
        directs=list(np.atleast_1d(direction))
        for d in range(np.size(directs)):
            if d==0:
                output_vec=self.__gaussianvelocity(coordinate=coordinate, direction=directs[d],sig=sig,echo=echo)
            elif d>0:
                output_vec=np.block([[output_vec],[self.__gaussianvelocity(coordinate=coordinate, direction=directs[d],sig=sig,echo=echo)]])
            else:
                raise ValueError('Wrong velocity direction list')
        return output_vec
        

class Rosenbrock_sys(EigenAnalysis):
    def __init__(self, mesh, boundary, omega=None, sigma=None,s=None, nu=None, path=None, baseflow=None):
        EigenAnalysis.__init__(self, mesh, boundary, nu=nu, path=path,baseflow=baseflow)
        self.IO_vec=InputOutput(mesh=mesh, boundary=boundary, omega=self.omega, sigma=self.sigma, nu=self.nu, element=element)
        
    def __Input(self, options):
        self.input_vec = eval("self.IO_vec."+self.IO_vec.dict_inputs[options['Variable']])

    def __Output(self, options):
        self.output_vec = eval("self.IO_vec."+self.IO_vec.dict_outputs[options['Variable']])
    
    def system(self):
        pass
        