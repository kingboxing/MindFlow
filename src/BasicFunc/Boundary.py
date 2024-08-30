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
    Select Boundary Condition
    """
    def __init__(self, element):
        self.bc_list={} # a dict of BCs
        self.element=element
    def VelocityInlet(self, mark, vel):
        """
        

        Parameters
        ----------
        mark : int
            DESCRIPTION.
        vel : tuple
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(mark) is int:
            if type(vel) is tuple:
                self.bc_list[mark].update({'FunctionSpace': 'V', 'Value': Constant(vel), 'BoundaryTraction': None})
            elif isinstance(vel, (function.function.Function, function.expression.Expression)):
                self.bc_list[mark].update({'FunctionSpace': 'V', 'Value': vel, 'BoundaryTraction': None})
            else:
                raise ValueError('Parameter type error of velocity-inlet boundary condition')
        else:
            raise ValueError('Parameter type error of velocity-inlet boundary condition')
        
        if self.element.type=='TaylorHood':
            self.bc_list[mark]['FunctionSpace']+='.sub(0)'
            
         
    def SlipWall(self, mark, norm, vel=0.0):
        """
        boundary parallel or normal to axis

        Parameters
        ----------
        mark : TYPE
            DESCRIPTION.
        norm : TYPE
            positive vector normal to the boundary.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(mark) is int and type(norm) is tuple:
            ind=str(norm.index(1))
            self.bc_list[mark].update({'FunctionSpace': 'V.sub('+ind+')',   'Value': Constant(vel), 'BoundaryTraction': None})
        else:
            raise ValueError('Parameter type error of symmetry boundary condition')
            
        if self.element.type=='TaylorHood':
            self.bc_list[mark]['FunctionSpace']=self.bc_list[mark]['FunctionSpace'].replace('V','V.sub(0)')
            
    
    def Symmetry(self, mark, norm):
        """
        

        Parameters
        ----------
        mark : TYPE
            DESCRIPTION.
        norm : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(mark) is int and type(norm) is tuple:
            self.SlipWall(mark, norm)
        else:
            raise ValueError('Parameter type error of slip wall boundary condition')
    
    def NoSlipWall(self, mark):
        """
        

        Parameters
        ----------
        mark : TYPE
            DESCRIPTION.
        dim : TYPE, optional
            DESCRIPTION. The default is 2.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        vel=(0,)*self.element.dimension
        
        if type(mark) is int:
            self.VelocityInlet(mark, vel)
        else:
            raise ValueError('Parameter type error of no-slip wall boundary condition')
    
    def FreeBoundary(self, mark):
        """
        

        Parameters
        ----------
        mark : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(mark) is int:
            self.bc_list[mark].update({'FunctionSpace': None,   'Value': 'Free Boundary', 'BoundaryTraction': None})
        else:
            raise ValueError('Parameter type error of free boundary condition')
    
    def PressureInlet(self, mark, pre, mode=1):
        """
        pending for case

        Parameters
        ----------
        mark : TYPE
            DESCRIPTION.
        pre : TYPE
            DESCRIPTION.
        mode : TYPE, optional
            DESCRIPTION. The default is 1.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if isinstance(pre, (int, float)):
            self.bc_list[mark].update({'FunctionSpace': 'Q',   'Value': Constant(pre), 'BoundaryTraction': (mark, mode)}) # mode 1 for BoundaryTraction 
        elif isinstance(pre, (function.function.Function, function.expression.Expression)):
            self.bc_list[mark].update({'FunctionSpace': 'Q',   'Value': pre, 'BoundaryTraction': mode})
        else:
            raise ValueError('Parameter type error of pressure inlet boundary condition')

        if self.element.type=='TaylorHood':
            self.bc_list[mark]['FunctionSpace']=self.bc_list[mark]['FunctionSpace'].replace('Q','Q.sub(1)')
          


    def PressureOutlet(self, mark, pre, mode=1):
        """
        pending for case

        Parameters
        ----------
        mark : TYPE
            DESCRIPTION.
        pre : TYPE
            DESCRIPTION.
        mode : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        if isinstance(pre, (int, float,function.function.Function, function.expression.Expression)):
            self.PressureInlet(mark, pre, mode=mode)
        else:
            raise ValueError('Parameter type error of pressure outlet boundary condition')
#%%

class SetBoundary(SubDomain, BoundaryCondition):
    """
    A subclass of SubDomain for defining and marking a boundary

    Parameters
    -------------------
    mesh : object created by FEniCS function Mesh

    mark : int
        Initial mark of the whole domain, default : 0

    Attributes
    -------------------
    boundary : FacetFunction with marked boundary on given mesh

    Examples
    -------------------
    >>> from MindFlow.BasicFunc.Boundary import SetBoundary
    >>> from dolfin import *
    >>> mesh = Mesh("mesh.xml")
    >>> BoundaryLocation = 'on_boundary and near(x[0], 0.0, tol)'
    >>> boundary=SetBoundary(mesh)
    >>> boundary.set_boundary(location = BoundaryLocations, mark = 1)

    """

    def __init__(self, mesh, element, mark_all=0):
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
        Function that returns True for points that belong to the boundary
        and False for points that don't belong to the boundary.
        """
        tol = self.tol
        return eval(self.option)

    def set_boundary(self, location, mark, tol=1e-10):
        """
        Function to define and mark the boundary

        Parameters
        ------------------
        location: string
            Indicate the location of the boundary

        mark: int
            The number that represents the boundary

        tol: float
            Tolerance while find the boundary, default : 1e-10

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
            else:
                pass
            
            # unused functionality
            self.mark(self.subboundary, mark)

    def get_measure(self):
        """
        Get the measure object of the domain

        Returns
        --------------------
        ds : a Measure object with id of boundaries in the domain
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

    Examples
    -------------------
    >>> from MindFlow.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
    >>> from dolfin import *
    >>> mesh = Mesh("mesh.xml")
    >>> P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    >>> P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    >>> TH = MixedElement([P2, P1])
    >>> V = FunctionSpace(mesh, TH)
    >>> boundarylocation = 'on_boundary and near(x[0], 0.0, tol)'
    >>> boundarycondition = {'FunctionSpace': 'V.sub(0).sub(0)', 'Value': Constant(0.0)}
    >>> boundary=SetBoundary(mesh)
    >>> boundary.set_boundary(location = boundarylocations, mark = 1)
    >>> bc = SetBoundaryCondition(functionspace=V, boundary=boundary)
    >>> bc.set_boundarycondition(boundarycondition, 1)

    """
    def __init__(self, functionspace, set_boundary):
        self.functionspace = functionspace # 
        self.set_boundary = set_boundary
        self.boundary = set_boundary.get_domain() # Get the FacetFunction on given mesh
        self.bc_list = [] # list of DirichletBC object
        self.has_free_bc=False
        
        self.v = TestFunction(self.functionspace)
        self.u = TrialFunction(self.functionspace)
        self.func = Function(self.functionspace)
        
    def set_boundarycondition(self, bc_dict, mark):
        """
        Set a boundary condition using FEniCS function DirichletBC as
            DirichletBC(self.functionspace.sub(0).sub(0), self.boundary, mark)

        Parameters
        ------------------------
        bc_dict: dict
            At least two keys : 'FunctionSpace' and 'Value', which respectively
            indicate the location and the value of the boundary condition

        mark: int
            ID of the boundary

        """
        if 'Value' not in bc_dict:
            if 'FunctionSpace' not in bc_dict:
                info('Please specify the FunctionSpace and Value of the Dirichlet Boundary Condition Applied at Boundary % g' % mark)
            else:
                info('Please specify the Value of the Dirichlet Boundary Condition Applied at Boundary % g' % mark)
        elif bc_dict['Value'] in ['Free Boundary']:
            info('Free boundary condition (zero boundary traction) applied at Boundary % g' % mark)
            self.has_free_bc+=1
        elif 'FunctionSpace' not in bc_dict:
            info('Please specify the FunctionSpace of the Dirichlet Boundary Condition Applied at Boundary % g' % mark)
        elif bc_dict['FunctionSpace'] is not None and bc_dict['Value'] is not None:
                index = bc_dict['FunctionSpace'].find('.') # find the index of the first dot
                if bc_dict['FunctionSpace'][index] == '.': # if bc_dict applied to a subspace
                    bc = 'DirichletBC(self.functionspace' + bc_dict['FunctionSpace'][index:] \
                        + ',' + "bc_dict['Value']" + ',' + 'self.boundary' + ',' + 'mark, method="geometric")' 
                else: # bc_dict applied to functionspace
                    bc= 'DirichletBC(self.functionspace' +  ',' + "bc_dict['Value']" + ',' \
                        + 'self.boundary' + ',' + 'mark, method="geometric")'
                self.bc_list.append(eval(bc)) # boundary condition added to the list bcs
        else:
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)

        # deal with 'FreeOutlet' BC
    
    def MatrixBC_rhs(self): # try with assemble module
        """
        Matrix that contains only zeros in the rows which have Dirichlet boundary conditions and ones in other rows
        """
        I = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
        I.ident_zeros()
        Mat_bc = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
        [bc.apply(Mat_bc) for bc in self.bc_list]
        Mat = I - Mat_bc
        return Mat
    
    def VectorBC_rhs(self): 
        """
        Vector that contains boundary condition values in the rows which have Dirichlet boundary conditions
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