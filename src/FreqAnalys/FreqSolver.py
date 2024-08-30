#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:32:49 2024

@author: bojin
"""

from src.Deps import *

from src.NSolver.SolverBase import NSolverBase
from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from src.BasicFunc.InitialCondition import SetInitialCondition
from src.Eqns.NavierStokes import Incompressible
from src.LinAlg.MatrixOps import AssembleMatrix, AssembleVector, InverseMatrixOperator, SparseLUSolver
from src.LinAlg.Utils import allclose_spmat, get_subspace_info

#%%
class FrequencyResponse(NSolverBase):
    """
    Solve Frequency Response of a linearized Navier-Stokes system with given input and output vectors.
    """
    def __init__(self, mesh, Re=None, order=(2,1), dim=2, constrained_domain=None):
        """
        Initialize the FrequencyResponse solver.

        Parameters
        ----------
        mesh : Mesh
            The mesh of the flow field.
        Re : float, optional
            Reynolds number. Default is None.
        order : tuple, optional
            Order of finite element. Default is (2,1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : SubDomain, optional
            Constrained domain defined in FEniCS. Default is None.

        Returns
        -------
        None.

        """
        element = TaylorHood(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        super().__init__(mesh, element, Re, None, None)
       
        # boundary condition
        self.boundary_condition = SetBoundaryCondition(self.element.functionspace, self.boundary)
        # init param
        self.param['solver_type']='frequency_solver'
        self.param['frequency_solver']={'method': 'lu', 
                                        'lusolver': 'mumps'}
        
    def set_baseflow(self, ic, timestamp=0.0):
        """
        Set the base flow around which Navier-Stokes equations are linearized.

        Parameters
        ----------
        ic : str or Function
            Path or FEniCS function stores the base flow.
        timestamp : float, optional
            Timestamp of the base flow saved in the time-series file if ic is a path. Default is 0.0.

        Returns
        -------
        None.

        """
        
        SetInitialCondition(0, ic=ic, fw=self.eqn.fw[0], timestamp=timestamp)
        
        
    def _form_LNS_equations(self, s):
        """
        Form the UFL expression of a linearized Navier-Stokes system.

        Parameters
        ----------
        s : int, float, complex
            The Laplace variable s, also known as the operator variable in the Laplace domain.

        Returns
        -------
        None.

        """
        
        # form Steady Linearised Incompressible Navier-Stokes Equations
        leqn=self.eqn.SteadyLinear()
        feqn=self.eqn.Frequency(s)
        
        for key in self.has_traction_bc.keys():
            leqn += self.BoundaryTraction(self.eqn.tp, self.eqn.tu, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
        
        self.LNS=(leqn+feqn[0], feqn[1]) # (real part, imag part)

    def _assemble_LHS(self, Mat=None):
        """
        Assemble the left-hand side [LHS=(sI-A)] of the linear system.
        
        u=(sI-A)^-1*f where RHS=f, Resp=u
        
        Parameters
        ----------
        Mat : scipy.sparse matrix, optional
            Feedback matrix (if control is applied). The default is None.

        Returns
        -------
        None.

        """
        # assemble the real part
        Ar = AssembleMatrix(self.LNS[0],self.boundary_condition.bc_list)
        
        if Mat is not None:# feedback loop
            Ar += Mat
            
        # assemble the imag part
        Ai = AssembleMatrix(self.LNS[1],self.boundary_condition.bc_list) 
        # matrix has only ones in diagonal for rows specified by the boundary condition
        I_bc = AssembleMatrix(Constant(0.0)*self.LNS[1],self.boundary_condition.bc_list)
        
        # Complex matrix L with boundary conditions
        LHS = Ar + Ai.multiply(1j)-I_bc.multiply(1j)
        # Convert to CSC format for efficient LU decomposition
        self.LHS = LHS.tocsc()
    
    def __initialize_solver(self, method = 'lu', lusolver='mumps'):
        """
        Initialize the Inverse Matrix Operator and LU solver.

        Parameters
        ----------
        method : str, optional
            Solve Method to use. Default is 'lu'.
        lusolver : str, optional
            Type of LU solver to use. Default is 'mumps'.

        Returns
        -------
        None.

        """
        if method == 'lu':
            info(f'LU decomposition using {lusolver.upper()} solver...')
            self.Linv=InverseMatrixOperator(self.LHS,lusolver=lusolver, echo=False)
            info('Done.')
            
        elif method == 'krylov':
            #self.Minv=precondition_jacobi(L, useUmfpack=useUmfpack)
            pass  # Krylov solver implementation pending
        
    def _check_vec(self, b):
        """
        Validate and flatten the 1-D vector for solving.

        Parameters
        ----------
        b : numpy array
            The vector to be validated. Should be a 1-D numpy array.

        Raises
        ------
        ValueError
            The shape of the array.

        Returns
        -------
        b : numpy array
            Flattened 1D array.
            
        """
        
        if np.size(b.shape)==1 or np.min(b.shape)==1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give 1D input/output array')
        
        return b
    
    def _solve_linear_system(self, input_vec, output_vec=None):
        """
        Solve the linearized system.

        Parameters
        ----------
        input_vec : numpy array
            Input vector for the system.
        output_vec : numpy array, optional
            Output vector for the system. Default is None (return flow state).

        Returns
        -------
        numpy array
            Frequency response vector (actuated by input_vec and measured by output_vec).

        """
        iv=self._check_vec(input_vec)
        self.state = self.Linv.matvec(iv)
        
        if output_vec is None:
            return self.state.reshape(-1, 1)  # Reshape to column vector
        else:
            ov=self._check_vec(output_vec).reshape(1, -1) # Reshape to row vector
            return ov @ self.state  # Matrix multiplication (equivalent to * and np.matrix)
    
    def solve(self, s=None, input_vec=None, output_vec=None, Re=None, Mat=None):
        """
        Solve the frequency response of the linearized Navier-Stokes system.

        Parameters
        ----------
        s : int, float, complex, optional
            The Laplace variable s, also known as the operator variable in the Laplace domain.
        input_vec : numpy array, optional
            1-D input vector. Default is None.
        output_vec : numpy array, optional
            1-D output vector. Default is None.
        Re : float, optional
            Reynolds number. Default is None.
        Mat : scipy.sparse matrix, optional
            Feedback matrix (if control is applied). Default is None.

        Returns
        -------
        None.

        """
        
        rebuild=False
        
        if Re is not None and (self.eqn.Re is None or not np.allclose(self.eqn.Re, Re, atol=1e-12)):
            # if not yet set Re or the current Re has a value different from the previous on
            self.eqn.Re = Re
            rebuild=True
        else:
            # if Re is None, raise error
            if self.eqn.Re is None:
                raise ValueError('Please indicate a Reynolds number')
            # else use previous Re (Re = None means reuse Reynolds number)
            
        if s is not None and (not hasattr(self.eqn, 's') or not np.allclose(self.eqn.s, s, atol=1e-12)):
            # if not yet set s or the current s has a value different from the previous one
            rebuild=True
        else:
            # if s not yet set and the current one is None, raise error
            if not hasattr(self.eqn, 's'):
                raise ValueError('Please indicate value of the Laplace variable s')
            # else use previous s (s = None means resue the value)
               
            
        if Mat is not None and (not hasattr(self, 'Mat') or self.Mat is None or not allclose_spmat(self.Mat, Mat, atol=1e-12)):
            # if self.Mat doesn't exist (1st solve) or self.Mat (previous Mat) is not equal to current Mat
            self.Mat=Mat
            rebuild=True
        else:
            if not (hasattr(self, 'Mat') and self.Mat is None):
                self.Mat=Mat
                rebuild=True
            #else: current Mat is None and the previous Mat is None (not 1st solve), do nothing; otherwise do above command


        if self.rebuild:
            self._form_LNS_equations(s)
            self._assemble_LHS(Mat)
            self.__initialize_solver(self.param['frequency_solver']['method'], self.param['frequency_solver']['lusolver'])
        
        self.gain=self._solve_linear_system(input_vec, output_vec)
        
#%%
class VectorGenerator:
    """
    A factory class to generate common input and output vectors for the FrequencyResponse class.
    """
    def __init__(self, element, bc_obj=None):
        """
        vector factory for frequency response solver and other applications
        
        Parameters
        ----------
        element : TYPE
            DESCRIPTION.
        bc_obj : SetBoundaryCondition: object, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.element=elelment
        self.boundary_condition = bc_obj
        if bc_obj is not None:
            self.bc_list = self.boundary_condition.bc_list
            self._assemble_bcs()
            
    
    def _assemble_vec(self, expr, test_func, bc_list=[]):
        """
        Assemble an expression or function into a vector (variational form)

        Parameters
        ----------
        expr : expression or function
            DESCRIPTION.
        test_func : test function
            test function from variational formula
        bc_list : list, optional
            boundary conditions. The default is [].
        Returns
        -------
        numpy array
            DESCRIPTION.

        """
        
        var_form = dot(expr, test_func) * dx
        vec_wgt = AssembleVector(var_form, bc_list)
        return np.asarray(vec_wgt).flatten()
    
    def _assemble_mat(self, trial_func, test_func, bc_list=[]):
        """
        Assemble an unknown function into a matrix (variational form)

        Parameters
        ----------
        trial_func : trialfunction
            trial function from variational formula
        test_func : test function
            test function from variational formula
        bc_list : boundary, optional
            boundary conditions. The default is [].

        Returns
        -------
        scipy.sparse.csr_matrix
            DESCRIPTION.

        """

        var_form = dot(trial_func, test_func) * dx
        mat_wgt = AssembleMatrix(var_form, bc_list)
        return mat_wgt # scipy.sparse.csr_matrix
        
    def _assemble_bcs(self):
        """
        Assemble matrix and vector for boundary conditions

        Returns
        -------
        None.

        """
        self.bc_mat = self.boundary_condition.MatrixBC_rhs()
        self.bc_vec = self.boundary_condition.VectorBC_rhs()
        
    def set_boundarycondition(self, vec):
        """
        Apply boundary conditions to a RHS vector (consider to mutiply variational weighted matrix first)

        Parameters
        ----------
        vec : numpy array
            DESCRIPTION.

        Returns
        -------
        numpy array
            DESCRIPTION.

        """
        vec_bc = self.bc_mat * ConvertVector(vec,flag='Vec2PETSc') + self.bc_vec
        
        return vec_bc.get_local()
        
    def variational_weight(self, vec):
        """
        multiply a vector by the variational weight matrix for linear problems

        Parameters
        ----------
        vec : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not hasattr(self, 'Mat_wgt'):
            self.Mat_wgt = self._assemble_mat(self.element.tw, self.element.tew)
            
        vec_wgt = ConvertMatrix(self.Mat_wgt, flag='Mat2PETSc') * ConvertVector(vec,flag='Vec2PETSc')
        return vec_wgt.get_local()
    

    # weighted vector with bcs
    # def gaussian_vector(self, center, sigma, scale = 1.0, index = 0):
        
    #     sub_spaces = get_subspace_info(self.element.functionspace)
    #     sub_inedex = find_subspace_index(index, sub_spaces)
        
    #     v = self.element.tew
    #     for i in sub_index:
    #         v = v[i]
        
    #     order = self.element.order[sub_inedex[0]]
    #     expr= self._gaussian_expr(center, sigma, scale, order)
        
    #     return self._assemble(expr, v) # weighted vector
    
    # non-weighted vector, easy for plot/check
    def _gaussian_expr(self, center, sigma, scale = 1.0, order = None):
        """
        

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.
        scale : TYPE, optional
            DESCRIPTION. The default is 1.0.
        order : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        expr : TYPE
            DESCRIPTION.

        """
        if order is None:
            order = self.element.order[0]
        dim = self.element.dimension 

        expr0 = 'scale * pow(pow(2.0*pi, dim/2) * pow(sigma, dim), -1)' # general formula
        expr1 = 'pow(-2.0*pow(sigma, 2),-1)'
        expr2 = '('
        for i in range(self.dim):
            expr2 += 'pow(x['+str(i)+']-center['+str(i)+'],2)'
            if i < self.dim - 1:
                expr2 += '+'
        expr2 += ')'
        
        expr = Expression(expr0+'*exp('+expr1+'*'+expr2+')', degree=order, scale = scale,sigma = sigma, center = center, dim = dim)
        return expr
    
    def gaussian_vector(self, center, sigma, scale = 1.0, index = 0, limit = None):

        """
        Generate a vector with a Gaussian distribution with a specified scale at given coordinates to a specific subspace of a mixed element.

        Parameters
        ----------
        center : tuple
            coordinate.
        sigma : float
            The standard deviation of the distribution.
        scale : float, optional
            Coefficient. The default is 1.0.
        index : int, optional
            scalar subspace index. The default is 0.
        limit : int, float, optional
            Vertices that are more than limit*sigma away from the center have a value of zero. The default is None.
        Returns
        -------
        numpy array
            DESCRIPTION.

        """
        # get subspace info
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_inedex = find_subspace_index(index, sub_spaces)
        # Access the subspace
        subfunc = self.functionspace
        w = self.element.w
        for i in sub_index:
            subfunc = subfunc.sub(i)
            w = w.sub(i)
        # form the expression
        order = self.element.order[sub_inedex[0]]
        expr= self._gaussian_expr(center, sigma, scale, order)
        # Interpolate the expression into the subspace
        func=interpolate(expr, subfunc.collapse())
        # Assign the interpolated function to the correct subspace of the mixed function
        assign(w, func)
        
        if limit is not None:
            # Convert the point coordinates to a Point object
            point = Point(*center)
            # get coordinates of global dofs
            dofs_coords = self.element.functionspace.tabulate_dof_coordinates()
            # get indices of subspace in global dofs and the corresponding coordinates
            subdofs_index = subfunc.dofmap().dofs() 
            vertex_coords = dofs_coords[subdofs_index, :]
            # find the global vertex index outside the limtation
            vertex_index = subdofs_index[np.asarray([point.distance(Point(*vertex)) for vertex in vertex_coords])>limit*sigma]
            # Set zero at the choosen vertex 
            self.element.w.vector()[vertex_index] = 0

        return self.element.w.vector().get_local()
        
    
    def point_vector(self, coord, index, scale = 1.0):
        """
        Apply a point source with a specified value at given coordinates to a specific subspace of a mixed element.

        Parameters
        ----------
        coord : tuple
            A tuple (x, y) representing the coordinates of the point.
        index : int
            The index of the scalar subspace to which the point source should be applied.
        scale : float, optional
            The value to apply at the specified point (default is 1.0).
            
        Returns
        -------
        numpy array
            1-D numpy array.

        """
        # get subspace info
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_inedex = find_subspace_index(index, sub_spaces)
        
        # Access the subspace
        w = self.element.w
        subfunc = self.element.functionspace
        for i in sub_index:
            subfunc = subfunc.sub(i)
        
        pt = Point(*coord)
        point_source = PointSource(subfunc, pt, scale)
        point_source.apply(w.vector())
        w.vector().apply("insert")
        
        return np.asarray(ConvertVector(w.vector())).flatten()
            
    def unit_vector(self, coord, index, scale=1.0):

        """
        Generate a unit vector with a `1` at the most closet coordinate and subspace.

        Parameters
        ----------
        coord : tuple
            the specified coordinate.
        index : int
            index of the scalar subspace
        scale : float, optional
            The value to apply at the specified point (default is 1.0).

        Returns
        -------
        vector : numpy array
            A unit vector of the specified length.
        coord_cloest : tuple
            The vertex coordinate that the value is applied (not exactly equals to the given coordinate).
        """
        # get subspace info
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_inedex = find_subspace_index(index, sub_spaces)
        # get coordinates of global dofs, sorted as in Mixed Element Function
        subfunc = self.element.functionspace
        dofs_coords = subfunc.tabulate_dof_coordinates()
        # get subsapce to apply
        w = self.element.w
        for i in sub_index:
            subfunc = subfunc.sub(i)
            w = w.sub(i)
        # get indices of subspace in global dofs and the corresponding coordinates
        subdofs_index = subfunc.dofmap().dofs() 
        vertex_coords = dofs_coords[subdofs_index, :]

        # Convert the point coordinates to a Point object
        point = Point(*coord)
        # find the closest global vertex index 
        closest_vertex_index = subdofs_index[np.argmin([point.distance(Point(*vertex)) for vertex in vertex_coords])]
        # closet coordinate (not exactly equals to the given coordinate)
        coord_cloest=dofs_coords[closest_vertex_index]
        # Set the value at the closest vertex in the specified subspace
        self.element.w.vector()[closest_vertex_index] = value

        return self.element.w.vector().get_local(), tuple(coord_cloest)
    
            
    def random_vector(self, distribution='uniform', seed=None):
        """
        Generate a random vector using a specified distribution.

        Parameters
        ----------
        distribution : str, optional
            Type of distribution to use ('uniform', 'normal'). Default is 'uniform'.
        seed : int, optional
            Seed for the random number generator. Default is None.

        Returns
        -------
        numpy array
            A random vector of the specified length.
        """
        vector_length = self.element.functionspace.dim()
        
        if seed is not None:
            np.random.seed(seed)
        
        if distribution == 'uniform':
            return np.random.rand(vector_length)
        elif distribution == 'normal':
            return np.random.randn(vector_length)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")
    
    # already weighted due to the physical nature
    def _force_expr(self, mark, Re):
        """
        body force expression

        Parameters
        ----------
        mark : TYPE
            DESCRIPTION.
        Re : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        eqn=Incompressible(self.element, self.boundary_condition.set_boundary, Re)
        force_expr=eqn.force_expr()
        
        dim = self.element.dimension

        force = ()
        for i in range(dim):
            temp = (assemble((force_expr[0][i]) * eqn.ds(mark)), assemble((force_expr[1][i]) * eqn.ds(mark)))
            force += (temp,) 
            
        return force # 1st index for direction, 2nd index for component
    
    def force_vector(self, mark=None, dirc=None, comp=None, Re=None, reuse=True):
        """
        evaluate the force on the body (lift or drag)

        Parameters
        ----------
        mark : int
            the boundary mark of the body. The default is None.
        dirc : int
            0 means X direction and 1 means Y direction. The default is None.
        comp : None or int
            0 means pressure part, 1 means stress part, None means the summation
            The default is None.
        Re : None or float
            Reynolds number. The default is None.
        reuse : bool
            if re-assemble the force expression. The default is False.

        Returns
        -------
        numpy array
            numpy array that can compute force by multipling the flow state vector.

        """

        # force act on the body
        if reuse is False or not hasattr(self, 'force'):
            self.force = self._force_expr(mark, Re)
        
        if comp is None:
            return self.force[dirc][0].get_local()+self.force[dirc][1].get_local()
        else:
            return self.force[dirc][comp].get_loca()
        
    def unit_oscaccel_vector(self, dirc, s):
        """
        Fluid: specify oscillating acceleration
        a = mag * e^(st) --> uniform force
        v = mag/s * e^(st) -->  boundary condition
        
        mag = 1.0 in the function, note aslo set 1.0 in the boundary conditions        
        Parameters
        ----------
        dirc : int
            direction of fluid's oscillation. index of scalar subspace
        s : int, float, complex
            The Laplace variable s, also known as the operator variable in the Laplace domain.

        Returns
        -------
        None.

        """
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_inedex = find_subspace_index(dirc, sub_spaces)
        
        tew = self.element.tew
        for i in sub_index:
            tew = tew.sub(i)
        
        vec = self._assemble_vec(Constant(1.0), tew, self.bc_list) + (1.0/s - 1.0) * self._assemble_vec(Constant(0.0), tew, self.bc_list)
                                                                            # set -1 at bc, and set zeros at rest
              # set 1 at bc and weighted uniform accel at rest       # set vel bc as 1/s, zero at rest
        
        return vec
    
    def unit_oscvel_vector(self, dirc, s):
        """
        Fluid: specify oscillating velocity 
        v = mag * e^(st) -->  boundary condition
        a = mag * s * e^(st) --> uniform force
        
        mag = 1.0 in the function, note aslo set 1.0 in the boundary conditions  
        Parameters
        ----------
        dirc : int
            direction of fluid's oscillation.
        s : int, float, complex
            The Laplace variable s, also known as the operator variable in the Laplace domain.

        Returns
        -------
        None.

        """
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_inedex = find_subspace_index(dirc, sub_spaces)
        
        tew = self.element.tew
        for i in sub_index:
            tew = tew.sub(i)
        
        vec = s * self._assemble_vec(Constant(1.0), tew, self.bc_list) + ( 1.0 - s) *self._assemble_vec(Constant(0.0), tew, self.bc_list)
        
        return vec
    
    def unit_oscdisp_vector(self, dirc, s):
        """
        Fluid: specify oscillating displacement
        d = mag * e^(st)
        v = mag * s * e^(st) -->  boundary condition
        a = mag * s^2 * e^(st) --> uniform force

        mag = 1.0 in the function, note aslo set 1.0 in the boundary conditions  
        Parameters
        ----------
        dirc : int
            direction of fluid's oscillation.
        s : int, float, complex
            The Laplace variable s, also known as the operator variable in the Laplace domain.

        Returns
        -------
        None.

        """
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_inedex = find_subspace_index(dirc, sub_spaces)
        
        tew = self.element.tew
        for i in sub_index:
            tew = tew.sub(i)
            
        vec = s * s * self._assemble_vec(Constant(1.0), tew, self.bc_list) + ( s - s * s) *self._assemble_vec(Constant(0.0), tew, self.bc_list)
        
        return vec
    
    def unit_rotvel_vector(self, mark, s=None, radius=1.0, center=(0.0, 0.0)):
        """
        2D Cylinder at origin (0,0): specify rotating velocity 
        v = mag * e^(st) at surface
        no acceleration applied on the fluid
        
        mag = radius = 1 in default
        Parameters
        ----------
        mark : int
            the boundary mark of the body. 
        s : int, float, complex, optional
            The Laplace variable s, also known as the operator variable in the Laplace domain. Default is None.
        radius : TYPE, optional
            DESCRIPTION. The default is 1.0.
        center : TYPE, optional
            DESCRIPTION. The default is (0.0, 0.0).
        Returns
        -------
        vec : numpy array
            DESCRIPTION.

        """
        index = 0
        tew = self.element.tew.sub(index) 
        
        vel_rotate = Expression(('(x[1]-center[1])/radius', '(-x[0]+center[0])/radius'), degree=self.element.order[index], center=center, radius = radius)
        BC_rotate = DirichletBC(self.element.functionspace.sub(index), vel_rotate, self.boundary_condition.boundary, mark, method="geometric")
        bcs = [BC_rotate]
        
        vec = self._assemble_vec(Constant((0.0, 0.0)), tew, bcs)
        
        return vec
    
    def unit_rotaccel_vector(self, mark, s, radius=1.0, center=(0.0, 0.0)):
        """
        2D Cylinder at origin (0,0): specify rotating acceleration
        a = mag * e^(st) at surface
        v = mag * 1/s * e^(st) at surface
        no acceleration applied on the fluid
        
        mag = radius = 1 in default
        Parameters
        ----------
        mark : int
            the boundary mark of the body. 
        s : int, float, complex, optional
            The Laplace variable s, also known as the operator variable in the Laplace domain.
        radius : float, optional
            Radius of the cylinder. The default is 1.0.
        center : TYPE, optional
            Center of the cylinder. The default is (0.0, 0.0).
        Returns
        -------
        None.

        """
        index = 0
        tew = self.element.tew.sub(index) 
        
        vel_rotate = Expression(('(x[1]-center[1])/radius', '(-x[0]+center[0])/radius'), degree=self.element.order[index], center=center, radius = radius)

        BC_clean = DirichletBC(self.element.functionspace.sub(index), Constant((0.0, 0.0)), self.boundary_condition.boundary, mark, method="geometric")
        BC_rotate = DirichletBC(self.element.functionspace.sub(index), vel_rotate, self.boundary_condition.boundary, mark, method="geometric")
        
        bcs0 = self.bc_list + [BC_clean]
        bcsr = [BC_rotate]
        
        vec0 = self._assemble_vec(Constant((0.0, 0.0)), tew, bcs0) # bc at other boundary, set zero at body surface
        vec_r = 1.0/s * self._assemble_vec(Constant((0.0, 0.0)), tew, bcsr) # set bc at body surface
        
        return vec0+vec_r
    
    def unit_rotangle_vector(self, mark ,s, radius=1.0, center=(0.0, 0.0)):
        """
        2D Cylinder at origin (0,0): specify rotating angle
        angle = mag * e^(st)
        v = mag * s * e^(st) at surface
        a = mag * s^2 * e^(st) at surface
        no acceleration applied on the fluid
        
        mag = radius = 1 in default

        Parameters
        ----------
        mark : int
            the boundary mark of the body. 
        s : int, float, complex, optional
            The Laplace variable s, also known as the operator variable in the Laplace domain.
        radius : float, optional
            Radius of the cylinder. The default is 1.0.
        center : TYPE, optional
            Center of the cylinder. The default is (0.0, 0.0).
            
        Returns
        -------
        None.

        """
        index = 0
        tew = self.element.tew.sub(index) 
        
        vel_rotate = Expression(('(x[1]-center[1])/radius', '(-x[0]+center[0])/radius'), degree=self.element.order[index], center=center, radius = radius)

        BC_clean = DirichletBC(self.element.functionspace.sub(index), Constant((0.0, 0.0)), self.boundary_condition.boundary, mark, method="geometric")
        BC_rotate = DirichletBC(self.element.functionspace.sub(index), vel_rotate, self.boundary_condition.boundary, mark, method="geometric")
        
        bcs0 = self.bc_list + [BC_clean]
        bcsr = [BC_rotate]
        
        vec0 = self._assemble_vec(Constant((0.0, 0.0)), tew, bcs0) # bc at other boundary, set zero at body surface
        vec_r = s * self._assemble_vec(Constant((0.0, 0.0)), tew, bcsr) # set bc at body surface
        
        return vec0+vec_r
    
    