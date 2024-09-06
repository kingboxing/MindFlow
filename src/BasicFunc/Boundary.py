#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:06:27 2023

@author: bojin
"""

from src.Deps import *

"""
This class provides subclasses of FEniCS interface of
defining and marking boundaries 
"""
#%%
class BoundaryCondition:
    """
    A factory of boundary conditions for FEniCS simulations.
    """
    def __init__(self, element):
        """
        Initialize the boundary condition manager.

        Parameters
        ----------
        element : object
            The finite element used in the simulation.
        """
        
        self.bc_list={} # a dict of BCs
        self.element=element
        
    def _validate_mark_and_value(self, mark, value, value_types=(tuple, function.function.Function, function.expression.Expression)):
        """
        Validate the mark and value parameters for boundary conditions.

        Parameters
        ----------
        mark : int
            The identifier for the boundary.
        value : tuple, Function, or Expression
            The value to apply at the boundary.
        value_types : tuple, optional
            Types that the value can be. Default is (tuple, Function, Expression).

        Raises
        ------
        ValueError
            If mark or value are not of the expected type.
        """
        if not isinstance(mark, int):
            raise ValueError('Boundary mark must be an integer.')
        if not isinstance(value, value_types):
            raise ValueError(f'Value must be one of {value_types}.')
        
    def VelocityInlet(self, mark, vel):
        """
        Apply a velocity inlet boundary condition.

        Parameters
        ----------
        mark : int
            The identifier for the boundary.
        vel : tuple, Function, or Expression
            The velocity to apply at the boundary.
            
        Returns
        -------
        None.

        """
        self._validate_mark_and_value(mark, vel)
        self.bc_list[mark].update({'FunctionSpace': 'V', 'Value': Constant(vel) if isinstance(vel, tuple) else vel, 'BoundaryTraction': None})

        if self.element.type == 'TaylorHood':
            self.bc_list[mark]['FunctionSpace'] += '.sub(0)'
         
    def SlipWall(self, mark, norm, vel=0.0):
        """
        Apply a slip wall boundary condition. 
        Now only for boundary parallel or normal to axis

        Parameters
        ----------
        mark : int
            The identifier for the boundary.
        norm : tuple
            A positive vector normal to the boundary.
        vel : float, optional
            The velocity across/normal the boundary. Default is 0.0.

        """
        self._validate_mark_and_value(mark, norm, value_types=(tuple,))
        
        ind = str(norm.index(1))
        self.bc_list[mark].update({'FunctionSpace': f'V.sub({ind})', 'Value': Constant(vel), 'BoundaryTraction': None})

        if self.element.type == 'TaylorHood':
            self.bc_list[mark]['FunctionSpace'] = self.bc_list[mark]['FunctionSpace'].replace('V', 'V.sub(0)')

    
    def Symmetry(self, mark, norm):
        """
        Apply a symmetry boundary condition by reusing SlipWall.

        Parameters
        ----------
        mark : int
            The identifier for the boundary.
        norm : tuple
            A vector normal to the boundary.

        Returns
        -------
        None.

        """
        self.SlipWall(mark, norm)
    
    def NoSlipWall(self, mark):
        """
        Apply a no-slip wall boundary condition.

        Parameters
        ----------
        mark : int
            The identifier for the boundary.

        """
        vel=(0,)*self.element.dim
        self.VelocityInlet(mark, vel)
    
    def FreeBoundary(self, mark):
        """
        Apply a free boundary condition (zero boundary traction).

        Parameters
        ----------
        mark : int
            The identifier for the boundary.

        """
        self._validate_mark_and_value(mark, None, value_types=(type(None),))
        
        self.bc_list[mark].update({'FunctionSpace': None, 'Value': 'Free Boundary', 'BoundaryTraction': None})
    
    def PressureInlet(self, mark, pre, mode=1):
        """
        Apply a pressure inlet boundary condition.
        pending for case testing ...
        
        Parameters
        ----------
        mark : int
            The identifier for the boundary.
        pre : int, float, Function, or Expression
            The pressure to apply at the boundary.
        mode : int, optional
            The mode of application. Default is 1.


        """
        self._validate_mark_and_value(mark, pre, value_types=(int, float, function.function.Function, function.expression.Expression))
        
        self.bc_list[mark].update({'FunctionSpace': 'Q', 'Value': Constant(pre) if isinstance(pre, (int, float)) else pre, 'BoundaryTraction': (mark, mode)})# mode 1 for BoundaryTraction 

        if self.element.type == 'TaylorHood':
            self.bc_list[mark]['FunctionSpace'] = self.bc_list[mark]['FunctionSpace'].replace('Q', 'Q.sub(1)')

    def PressureOutlet(self, mark, pre, mode=1):
        """
        Apply a pressure outlet boundary condition using PressureInlet.
        pending for case testing ...

        Parameters
        ----------
        mark : int
            The identifier for the boundary.
        pre : int, float, Function, or Expression
            The pressure to apply at the boundary.
        mode : int, optional
            The mode of Boundary Traction type. Default is 1.

        Returns
        -------
        None.

        """
        self.PressureInlet(mark, pre, mode)

#%%

class SetBoundary(SubDomain, BoundaryCondition):
    """
    A subclass of SubDomain for defining and marking a boundary.

    Attributes
    -------------------
    boundary : FacetFunction with marked boundary on given mesh

    """

    def __init__(self, mesh, element, mark_all=0):
        """
        Initialize 
        
        Parameters
        ----------
        mesh : object created by FEniCS function Mesh
            DESCRIPTION.
        element : object
            The finite element used in the simulation.
        mark_all : int, optional
            Initial mark of the whole domain. The default is 0.

        Returns
        -------
        None.

        """
        SubDomain.__init__(self) # initialize base class
        BoundaryCondition.__init__(self, element)
        self.mark_all=mark_all
        self.mesh = mesh
        self.boundary = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)#FacetFunction("size_t", self.mesh); using int to mark boundaries
        self.boundary.set_all(self.mark_all) # mark all facets as 0 
                
        # unused functionality
        self.submesh=BoundaryMesh(mesh, 'exterior') # boundary mesh with outward pointing normals 
        self.subboundary=MeshFunction('size_t', self.submesh, self.submesh.topology().dim())
        self.subboundary.set_all(self.mark_all)

    def inside(self, x, on_boundary):
        """
        Determine if a point is inside the boundary.

        Parameters
        ----------
        x : array-like
            Coordinates of the point.
        on_boundary : bool
            Whether the point is on the boundary.

        Returns
        -------
        bool
            True if the point is inside the boundary, False otherwise.
        """
        tol = self.tol
        return eval(self.option)

    def set_boundary(self, location, mark, tol=1e-10):
        """
        Define and mark the boundary.

        Parameters
        ----------
        location : str
            The location of the boundary.
        mark : int
            The identifier for the boundary.
        tol : float, optional
            Tolerance for boundary location. Default is 1e-10.
        """

        self.tol = tol
        if mark == self.mark_all:
            info('Warning : The Mark Number of the Boundary is the Same as the Mesh Facets')
        else:
            self.option = location
            self.mark(self.boundary, mark)
            try:
                self.option = location.replace('on_boundary and ','')
            except:
                self.option = location.replace(' and on_boundary','')

            # unused functionality
            self.mark(self.subboundary, mark)

    def get_measure(self):
        """
        Get the measure object of the domain

        Returns
        -------
        Measure
            A Measure object with IDs of boundaries in the domain.
        """
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundary)
        return ds

    def get_domain(self):
        """
        Get the FacetFunction on given mesh

        Returns
        ---------------------
        boundary : FacetFunction with marked boundary on given mesh
        """
        return self.boundary
    
    # unused functionality
    def get_submeasure(self):
        ds = Measure('ds', domain=self.submesh, subdomain_data=self.subboundary)
        return ds
        
    def get_subdomain(self):
        return self.subboundary
    
#%%
"""
This class provides classes for setting up boundary conditions
"""

class SetBoundaryCondition:
    """
    A class to set up the Dirichlet boundary condition

    Parameters
    -------------------
    Functionspace : a finite element function space

    set_boundary : object SetBoundary()
        the SetBoundary object with defined and marked boundaries

    Attributes
    -------------------
    functionspace : a finite element function space

    boundary : FacetFunction on given mesh

    bc_list : a list with boundary conditions


    """
    def __init__(self, functionspace, set_boundary):
        """
        Initialise

        Parameters
        ----------
        functionspace : FunctionSpace
            a finite element function space.
        set_boundary : SetBoundary
            SetBoundary object with defined and marked boundaries.

        Returns
        -------
        None.

        """
        self.functionspace = functionspace # 
        self.set_boundary = set_boundary
        self.boundary = set_boundary.get_domain() # Get the FacetFunction on given mesh
        self.bc_list = [] # list of DirichletBC object
        self.has_free_bc=False
        
        #
        self.v = TestFunction(self.functionspace)
        self.u = TrialFunction(self.functionspace)
        self.func = Function(self.functionspace)
        
    def set_boundarycondition(self, bc_dict, mark):
        """
        Set a boundary condition using FEniCS' DirichletBC as
            DirichletBC(self.functionspace.sub(0).sub(0), self.boundary, mark)
        
        Parameters
        ------------------------
        bc_dict: dict
            Contains 'FunctionSpace' and 'Value' keys.
            which respectively indicate the subspace and the value of the boundary condition
        mark: int
            The identifier for the boundary.

        """
        if 'FunctionSpace' not in bc_dict or 'Value' not in bc_dict:
            raise ValueError(f'Missing FunctionSpace/Value for boundary {mark}')
        
        if bc_dict['Value'] in ['Free Boundary']: # if free bc, then do nothing
            info(f'Free boundary condition (zero boundary traction) applied at boundary {mark}')
            self.has_free_bc+=1
            return
        
        function_space_str = bc_dict['FunctionSpace'] # FunctionSpace
        value = "bc_dict['Value']" # Value in str
        if function_space_str is None or value is None:
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
            return
        
         # start apply DirichletBC
        index = function_space_str.find('.') # if FunctionSpace is a subspace
        if function_space_str[index] == '.': # if bc_dict applied to a subspace
            sub_function_space=function_space_str[index:]
        else: # bc_dict directly applied to functionspace
            sub_function_space=''
        
        func = f"DirichletBC(self.functionspace{sub_function_space}, {value}, self.boundary, mark, method='geometric')"
        self.bc_list.append(eval(func)) # boundary condition added to the list bcs


        # if 'Value' not in bc_dict:
        #     if 'FunctionSpace' not in bc_dict:
        #         info('Please specify the FunctionSpace and Value of the Dirichlet Boundary Condition Applied at Boundary % g' % mark)
        #     else:
        #         info('Please specify the Value of the Dirichlet Boundary Condition Applied at Boundary % g' % mark)
        # elif bc_dict['Value'] in ['Free Boundary']:
        #     info('Free boundary condition (zero boundary traction) applied at Boundary % g' % mark)
        #     self.has_free_bc+=1
        # elif 'FunctionSpace' not in bc_dict:
        #     info('Please specify the FunctionSpace of the Dirichlet Boundary Condition Applied at Boundary % g' % mark)
        # elif bc_dict['FunctionSpace'] is not None and bc_dict['Value'] is not None:
        #         index = bc_dict['FunctionSpace'].find('.') # find the index of the first dot
        #         if bc_dict['FunctionSpace'][index] == '.': # if bc_dict applied to a subspace
        #             bc = 'DirichletBC(self.functionspace' + bc_dict['FunctionSpace'][index:] \
        #                 + ',' + "bc_dict['Value']" + ',' + 'self.boundary' + ',' + 'mark, method="geometric")' 
        #         else: # bc_dict applied to functionspace
        #             bc= 'DirichletBC(self.functionspace' +  ',' + "bc_dict['Value']" + ',' \
        #                 + 'self.boundary' + ',' + 'mark, method="geometric")'
        #         self.bc_list.append(eval(bc)) # boundary condition added to the list bcs
        # else:
        #     info('No Dirichlet Boundary Condition at Boundary % g' % mark)

        # deal with 'FreeOutlet' BC
    
    def MatrixBC_rhs(self): # try with assemble module
        """
        Create a matrix with zeros in rows that have Dirichlet boundary conditions and ones in diagonal elsewhere.
        Returns
        -------
        PETScMatrix
            A matrix with applied boundary conditions.
        """
        I = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
        I.ident_zeros()
        Mat_bc = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
        [bc.apply(Mat_bc) for bc in self.bc_list]
        return I - Mat_bc
    
    def VectorBC_rhs(self): 
        """
        Create a vector with BC values in rows that have Dirichlet boundary conditions.
        Returns
        -------
        PETScVector
            A vector with applied boundary conditions.
        """        
        Vec_bc = assemble(Constant(0.0)*dot(self.func, self.v) * dx)
        [bc.apply(Vec_bc) for bc in self.bc_list]
        return Vec_bc
    

#%%

# class BoundaryConditionFormat:
#     """
#     boundary conditions of cases for testing
#     """
    
#     def __init__(self):
#         self.mesh_list=['cylinder_8k.xml','cylinder_13k.xml','cylinder_26k.xml','cylinder_74k_sym_60ds_40us.xml']
#         self.bc_list=['homogeneous','inhomogeneous']
        
#     def get_boundaryconditions(self,mesh_name='cylinder_26k.xml',bc_type='homogeneous'):
#         if mesh_name in self.mesh_list and bc_type in self.bc_list:
#             return eval('self.__'+mesh_name[0:-4]+"('"+bc_type+"')")
#         else:
#             raise ValueError('boundary conditions type is not in the default list')
    
#     def __cylinder_homogeneous(self):
#         BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top',     'Mark': 1},
#                             'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom',  'Mark': 2},
#                             'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet',   'Mark': 3},
#                             'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder','Mark': 5},
#                             'Outlet'  : {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet',  'Mark': 4}
#                             }
#         return BoundaryConditions
                   
#     def __cylinder_inhomogeneous(self):
#         BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top',     'Mark': 1},
#                             'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom',  'Mark': 2},
#                             'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((1.0,0.0)), 'Boundary': 'inlet',   'Mark': 3},
#                             'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder','Mark': 5},
#                             'Outlet'  : {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet',  'Mark': 4}
#                             }
#         return BoundaryConditions
        
#     def __cylinder_8k(self,bc_type):
#         return eval('self.__cylinder_'+bc_type+'()')
                   
#     def __cylinder_13k(self,bc_type):
#         return eval('self.__cylinder_'+bc_type+'()')
            
#     def __cylinder_26k(self,bc_type):
#         return eval('self.__cylinder_'+bc_type+'()')
                   
#     def __cylinder_74k_sym_60ds_40us(self,bc_type):
#         return eval('self.__cylinder_'+bc_type+'()')

#%%
# class BoundaryFormat:
#     """
#     boundaries of meshes for testing
#     """
#     def __init__(self):
#         self.mesh_list=['cylinder_8k.xml','cylinder_13k.xml','cylinder_26k.xml','cylinder_74k_sym_60ds_40us.xml']

#     def get_boundary(self,mesh_name='cylinder_26k.xml'):
#         if mesh_name in self.mesh_list:
#             return eval('self.__'+mesh_name[0:-4]+'()')
#         else:
#             raise ValueError('boundary type is not in the test list')
    
#     def __cylinder_8k(self):
#         BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 12.0, tol)'},
#                            'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -12.0, tol)'},
#                            'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
#                            'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 20.0, tol)'},
#                            'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
#                             }
#         return BoundaryLocations
                   
#     def __cylinder_13k(self):
#         BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
#                            'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
#                            'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
#                            'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
#                            'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
#                            }
#         return BoundaryLocations
            
#     def __cylinder_26k(self):
#         BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
#                            'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
#                            'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
#                            'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
#                            'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
#                             }
                            
#         return BoundaryLocations
                   
#     def __cylinder_74k_sym_60ds_40us(self):
#         BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 40.0, tol)'},
#                            'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -40.0, tol)'},
#                            'Inlet'      : {'Mark': 3, 'Location':'on_boundary and near(x[0], -60.0, tol)'},
#                            'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 60.0, tol)'},
#                            'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
#                            }
                           
#         return BoundaryLocations