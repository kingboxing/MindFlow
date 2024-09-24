#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:46:44 2024

@author: bojin

VectorAsm Module

This module provides classes for generating common input/output vectors for these solvers.
"""

from ..Deps import *

from ..Eqns.NavierStokes import Incompressible
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, ConvertVector,ConvertMatrix, InverseMatrixOperator
from ..LinAlg.Utils import allclose_spmat, get_subspace_info, find_subspace_index

#%%
class VectorGenerator:
    """
    A factory class to generate common input and output vectors for the FrequencyResponse class.
    """
    def __init__(self, element, bc_obj=None):
        """
        Initialize the VectorGenerator.

        Parameters
        ----------
        element : object
            The finite element object defining the solution space.
        bc_obj : SetBoundaryCondition, optional
            Boundary condition object. Default is None.
        """
        
        self.element=element#.new()
        self.boundary_condition = bc_obj
        if bc_obj is not None:
            self.bc_list = self.boundary_condition.bc_list
            self._assemble_bcs()
            
    
    def _assemble_vec(self, expr, test_func, bc_list=[]):
        """
        Assemble an expression or function into a vector (variational form).

        Parameters
        ----------
        expr : expression or function
            The expression or function to assemble.
        test_func : function
            Test function from variational formula.
        bc_list : list, optional
            Boundary conditions. Default is [].

        Returns
        -------
        numpy array
            Assembled vector as a numpy array.
        """
        
        var_form = dot(expr, test_func) * dx
        vec_wgt = AssembleVector(var_form, bc_list)
        return np.asarray(vec_wgt).flatten()
    
    def _assemble_mat(self, trial_func, test_func, bc_list=[]):
        """
        Assemble an unknown function into a matrix (variational form).

        Parameters
        ----------
        trial_func : function
            Trial function from variational formula.
        test_func : function
            Test function from variational formula.
        bc_list : list, optional
            Boundary conditions. Default is [].

        Returns
        -------
        scipy.sparse.csr_matrix
            Assembled matrix in CSR format.
        """
        

        var_form = dot(trial_func, test_func) * dx
        mat_wgt = AssembleMatrix(var_form, bc_list)
        return mat_wgt # scipy.sparse.csr_matrix
        
    def _assemble_bcs(self):
        """
        Assemble matrices and vectors for boundary conditions.
        """
        self.bc_mat = self.boundary_condition.MatrixBC_rhs()
        self.bc_vec = self.boundary_condition.VectorBC_rhs()
        
    def set_boundarycondition(self, vec):
        """
        Apply boundary conditions to a RHS vector. (consider to mutiply variational weighted matrix first)

        Parameters
        ----------
        vec : numpy array
            The vector to apply boundary conditions to.

        Returns
        -------
        numpy array
            Vector with boundary conditions applied.
        """
        vec_bc = self.bc_mat * ConvertVector(vec,flag='Vec2PETSc') + self.bc_vec
        
        return vec_bc.get_local()
        
    def variational_weight(self, vec):
        """
        Multiply a vector by the variational weight matrix for linear problems.

        Parameters
        ----------
        vec : numpy array
            Vector to be weighted.

        Returns
        -------
        numpy array
            Weighted vector.
        """
        
        if not hasattr(self, 'Mat_wgt'):
            self.Mat_wgt = self._assemble_mat(self.element.tw, self.element.tew)
            
        vec_wgt = ConvertMatrix(self.Mat_wgt, flag='Mat2PETSc') * ConvertVector(vec,flag='Vec2PETSc')
        return vec_wgt.get_local()
    

    # weighted vector with bcs
    # def gaussian_vector(self, center, sigma, scale = 1.0, index = 0):
        
    #     sub_spaces = get_subspace_info(self.element.functionspace)
    #     sub_index = find_subspace_index(index, sub_spaces)
        
    #     v = self.element.tew
    #     for i in sub_index:
    #         v = v[i]
        
    #     order = self.element.order[sub_index[0]]
    #     expr= self._gaussian_expr(center, sigma, scale, order)
        
    #     return self._assemble(expr, v) # weighted vector
    
    # non-weighted vector, easy for plot/check
    def _gaussian_expr(self, center, sigma, scale = 1.0, order = None):
        """
        Define the Gaussian expression used to generate a Gaussian vector.

        Parameters
        ----------
        center : tuple
            The center of the Gaussian distribution.
        sigma : float
            The standard deviation of the Gaussian distribution.
        scale : float, optional
            The scale factor for the Gaussian distribution. Default is 1.0.
        order : int, optional
            The order of the finite element space. Default is None.

        Returns
        -------
        Expression
            The Gaussian expression.
        """
        if order is None:
            order = self.element.order[0]
        dim = self.element.dim

        expr0 = 'scale * pow(pow(2.0*pi, dim/2) * pow(sigma, dim), -1)' # general formula
        expr1 = 'pow(-2.0*pow(sigma, 2),-1)'
        expr2 = '('
        for i in range(self.element.dim):
            expr2 += 'pow(x['+str(i)+']-'+str(center[i])+',2)'
            if i < self.element.dim - 1:
                expr2 += '+'
        expr2 += ')'
        expr = Expression(expr0+'*exp('+expr1+'*'+expr2+')', degree=order, scale = scale,sigma = sigma, dim = dim)
        return expr
    
    def gaussian_vector(self, center, sigma, scale = 1.0, index = 0, limit = None):

        """
        Generate a vector with a Gaussian distribution.

        Parameters
        ----------
        center : tuple
            Coordinates of the center of the Gaussian distribution.
        sigma : float
            Standard deviation of the distribution.
        scale : float, optional
            Scale factor for the distribution. Default is 1.0.
        index : int, optional
            Scalar subspace index. Default is 0.
        limit : float, optional
            When the distance from the center exceeds Limit * sigma, the value is set to 0. Default is None.

        Returns
        -------
        numpy array
            Generated Gaussian vector.
        """
        
        # get subspace info
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_index = find_subspace_index(index, sub_spaces)
        # Access the subspace
        subfunc = self.element.functionspace
        w = self.element.add_functions()
        w_sub = w
        for i in sub_index:
            subfunc = subfunc.sub(i)
            w_sub = w_sub.sub(i)
        # form the expression
        order = self.element.order[sub_index[0]]
        expr= self._gaussian_expr(center, sigma, scale, order)
        # Interpolate the expression into the subspace
        func=interpolate(expr, subfunc.collapse())
        # Assign the interpolated function to the correct subspace of the mixed function
        assign(w_sub, func)
        
        if limit is not None:
            # Convert the point coordinates to a Point object
            point = Point(*center)
            # get coordinates of global dofs
            dofs_coords = self.element.functionspace.tabulate_dof_coordinates()
            # get indices of subspace in global dofs and the corresponding coordinates
            subdofs_index = subfunc.dofmap().dofs() 
            vertex_coords = dofs_coords[subdofs_index, :]
            # find the global vertex index outside the limtation
            vertex_index = np.asarray(subdofs_index)[np.asarray([point.distance(Point(*vertex)) for vertex in vertex_coords])>limit*sigma]
            # Set zero at the choosen vertex 
            w.vector()[vertex_index] = 0

        return w.vector().get_local()
        
    
    def point_vector(self, coord, index, scale = 1.0):
        """
        Apply a point source at specified coordinates to a subspace of a mixed element.

        Parameters
        ----------
        coord : tuple
            Coordinates of the point.
        index : int
            Index of the scalar subspace.
        scale : float, optional
            Value to apply at the specified point. Default is 1.0.

        Returns
        -------
        numpy array
            Vector with the point source applied.
            np.sum(vec)=1
        """
        
        # get subspace info
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_index = find_subspace_index(index, sub_spaces)
        
        # Access the subspace
        w = self.element.add_functions()
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
        sub_index = find_subspace_index(index, sub_spaces)
        # get coordinates of global dofs, sorted as in Mixed Element Function
        subfunc = self.element.functionspace
        dofs_coords = subfunc.tabulate_dof_coordinates()
        # get subsapce to apply
        w = self.element.add_functions()
        for i in sub_index:
            subfunc = subfunc.sub(i)

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
        w.vector()[closest_vertex_index] = scale

        return w.vector().get_local(), tuple(coord_cloest)
    
            
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
        Define the force expression used to calculate the lift or drag.

        Parameters
        ----------
        mark : int
            the boundary mark of the body.
        Re : float
            Reynolds number.

        Returns
        -------
        Expression
            The force expression.
            1st index for direction, 2nd index for component

        """
        eqn=Incompressible(self.element, self.boundary_condition.set_boundary, Re)
        force_expr=eqn.force_expr()
        
        dim = self.element.dim

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
        Generate a vector for fluid oscillating acceleration.
        
        Fluid: specified oscillating acceleration
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
        numpy array
            Vector for fluid oscillating acceleration.

        """
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_index = find_subspace_index(dirc, sub_spaces)
        
        tew = self.element.tew
        for i in sub_index:
            tew = tew.sub(i)
        
        vec = self._assemble_vec(Constant(1.0), tew, self.bc_list) + (1.0/s - 1.0) * self._assemble_vec(Constant(0.0), tew, self.bc_list)
                                                                            # set -1 at bc, and set zeros at rest
              # set 1 at bc and weighted uniform accel at rest       # set vel bc as 1/s, zero at rest
        
        return vec
    
    def unit_oscvel_vector(self, dirc, s):
        """
        Generate a vector for fluid oscillating velocity.
        
        Fluid: specified oscillating velocity 
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
        numpy array
            Vector for fluid oscillating acceleration.

        """
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_index = find_subspace_index(dirc, sub_spaces)
        
        tew = self.element.tew
        for i in sub_index:
            tew = tew.sub(i)
        
        vec = s * self._assemble_vec(Constant(1.0), tew, self.bc_list) + ( 1.0 - s) *self._assemble_vec(Constant(0.0), tew, self.bc_list)
        
        return vec
    
    def unit_oscdisp_vector(self, dirc, s):
        """
        Generate a vector for fluid oscillating displacement.
        
        Fluid: specified oscillating displacement
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
        numpy array
            Vector for fluid oscillating acceleration.

        """
        sub_spaces = get_subspace_info(self.element.functionspace)
        sub_index = find_subspace_index(dirc, sub_spaces)
        
        tew = self.element.tew
        for i in sub_index:
            tew = tew.sub(i)
            
        vec = s * s * self._assemble_vec(Constant(1.0), tew, self.bc_list) + ( s - s * s) *self._assemble_vec(Constant(0.0), tew, self.bc_list)
        
        return vec
    
    def unit_rotvel_vector(self, mark, s=None, radius=1.0, center=(0.0, 0.0)):
        """
        Generate a vector for rotating velocity of a 2D cylinder at the origin.
        
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
            Vector for rotating velocity.

        """
        index = 0
        tew = self.element.tew.sub(index) 
        
        vel_rotate = Expression(('(x[1]-center[1])/radius', '(-x[0]+center[0])/radius'), 
                                degree=self.element.order[index], center=center, radius = radius)
        BC_rotate = DirichletBC(self.element.functionspace.sub(index), vel_rotate, self.boundary_condition.boundary, mark, method="geometric")
        
        vec = self._assemble_vec(Constant((0.0, 0.0)), tew, [BC_rotate])
        
        return vec
    
    def unit_rotaccel_vector(self, mark, s, radius=1.0, center=(0.0, 0.0)):
        """
        Generate a vector for rotating acceleration of a 2D cylinder at the origin.
        
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
        numpy array
            Vector for rotating acceleration.

        """
        index = 0
        tew = self.element.tew.sub(index) 
        
        vel_rotate = Expression(('(x[1]-center[1])/radius', '(-x[0]+center[0])/radius'), 
                                degree=self.element.order[index], center=center, radius = radius)

        BC_clean = DirichletBC(self.element.functionspace.sub(index), Constant((0.0, 0.0)), self.boundary_condition.boundary, mark, method="geometric")
        BC_rotate = DirichletBC(self.element.functionspace.sub(index), vel_rotate, self.boundary_condition.boundary, mark, method="geometric")
        
        bcs0 = self.bc_list + [BC_clean]
        bcsr = [BC_rotate]
        
        vec0 = self._assemble_vec(Constant((0.0, 0.0)), tew, bcs0) # bc at other boundary, set zero at body surface
        vec_r = 1.0/s * self._assemble_vec(Constant((0.0, 0.0)), tew, bcsr) # set bc at body surface
        
        return vec0+vec_r
    
    def unit_rotangle_vector(self, mark ,s, radius=1.0, center=(0.0, 0.0)):
        """
        Generate a vector for rotating angle of a 2D cylinder at the origin.
        
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
        numpy array
            Vector for rotating angle.

        """
        index = 0
        tew = self.element.tew.sub(index) 
        
        vel_rotate = Expression(('(x[1]-center[1])/radius', '(-x[0]+center[0])/radius'), 
                                degree=self.element.order[index], center=center, radius = radius)

        BC_clean = DirichletBC(self.element.functionspace.sub(index), Constant((0.0, 0.0)), self.boundary_condition.boundary, mark, method="geometric")
        BC_rotate = DirichletBC(self.element.functionspace.sub(index), vel_rotate, self.boundary_condition.boundary, mark, method="geometric")
        
        bcs0 = self.bc_list + [BC_clean]
        bcsr = [BC_rotate]
        
        vec0 = self._assemble_vec(Constant((0.0, 0.0)), tew, bcs0) # bc at other boundary, set zero at body surface
        vec_r = s * self._assemble_vec(Constant((0.0, 0.0)), tew, bcsr) # set bc at body surface
        
        return vec0+vec_r
    
    