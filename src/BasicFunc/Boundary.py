#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:06:27 2023

@author: bojin
"""

from dolfin import *

"""
This module provides subclasses of FEniCS interface for
defining and marking boundaries 
"""


class SetBoundary(SubDomain):
    """A subclass of SubDomain for defining and marking a boundary

    Parameters
    -------------------
    mesh : object created by FEniCS function Mesh

    mark : int
        Initial mark of the whole domain, default : 0

    Attributes
    -------------------
    boundaries : FacetFunction with marked boundaries on given mesh

    Examples
    -------------------
    >>> from MindFlow.BasicFunc.Boundary import SetBoundary
    >>> from dolfin import *
    >>> mesh = Mesh("mesh.xml")
    >>> BoundaryLocation = 'on_boundary and near(x[0], 0.0, tol)'
    >>> boundary=SetBoundary(mesh)
    >>> boundary.set_boundary(location = BoundaryLocations, mark = 1)

    """

    def __init__(self, mesh=None, mark=0):
        SubDomain.__init__(self) # initialize base class
        if mesh is None:
            print('Error : Mesh Needed')
        else:
            self.mesh = mesh
            self.boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)#FacetFunction("size_t", self.mesh)
            self.boundaries.set_all(mark) # mark the whole domain as 0
            self.submesh=BoundaryMesh(mesh, 'exterior')
            self.subboundaries=MeshFunction('size_t', self.submesh, self.submesh.topology().dim())
            self.subboundaries.set_all(mark)

    def inside(self, x, on_boundary):
        """Function that returns True for points that belong to the boundary
            and False for points that don't belong to the boundary.
        """
        tol = self.tol
        return eval(self.option)

    def set_boundary(self, location=None, mark=None, tol=1e-4):
        """Function to define and mark the boundary

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
        if location is None or mark is None:
            print('Error : Please Indentify the Loacation or Mark of the Boundary')
        else:
            self.option = location
            self.mark(self.boundaries, mark)
            try:
                self.option = location.replace('on_boundary and ','')
            except:
                self.option = location.replace(' and on_boundary','')
            else:
                pass
            self.mark(self.subboundaries, mark)

    def get_measure(self):
        """Get the measure object of the domain

        Returns
        --------------------
        ds : a Measure object with id of boundaries in the domain
        """
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        return ds
        
    def get_submeasure(self):
        ds = Measure('ds', domain=self.submesh, subdomain_data=self.subboundaries)
        return ds

    def get_domain(self):
        """Get the FacetFunction on given mesh

        Returns
        ---------------------
        boundaries : FacetFunction with marked boundaries on given mesh
        """
        return self.boundaries
        
    def get_subdomain(self):
        return self.subboundaries
    
    
"""This module provides classes for setting up boundary conditions
"""


class SetBoundaryCondition:
    """A class to set up the Dirichlet boundary condition

    Parameters
    -------------------
    Functionspace : a finite element function space

    boundary : object Boundary()
        the Boundary object with defined and marked boundaries

    Attributes
    -------------------
    functionspace : a finite element function space

    boundaries : FacetFunction on given mesh

    bcs : a list with boundary conditions

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
    >>> bc = SetBoundaryCondition(Functionspace=V, boundary=boundary)
    >>> bc.set_boundarycondition(boundarycondition, 1)

    """
    def __init__(self, Functionspace=None, boundary=None):
        self.functionspace = Functionspace
        self.boundaries = boundary.get_domain() # Get the FacetFunction on given mesh
        self.bcs = []
        self.v = TestFunction(self.functionspace)
        self.u = TrialFunction(self.functionspace)
        self.func = Function(self.functionspace)
        
    def set_boundarycondition(self, boucon, mark):
        """Set a boundary condition using FEniCS function DirichletBC as
            DirichletBC(self.functionspace.sub(0).sub(0), self.boundaries, mark)

        Parameters
        ------------------------
        boucon: dict
            At least two elements : 'FunctionSpace' and 'Value', which respectively
            indicate the location and the value of the boundary condition

        mark: int
            ID of the boundary

        """
        ## test content in boucon
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
        
        if boucon['FunctionSpace'] is not None and boucon['Value'] is not None:
            index = boucon['FunctionSpace'].find('.') # find the index of the first dot
            if boucon['FunctionSpace'][index] == '.':
                bc = 'DirichletBC(self.functionspace' + boucon['FunctionSpace'][index:] \
                    + ',' + "boucon['Value']" + ',' + 'self.boundaries' + ',' + 'mark, method="geometric")'
            else:
                bc= 'DirichletBC(self.functionspace' +  ',' + "boucon['Value']" + ',' \
                    + 'self.boundaries' + ',' + 'mark, method="geometric")'
            self.bcs.append(eval(bc)) # boundary condition added to the list bcs
        else:
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
    
    def MatrixBC_rhs(self):
        """Matrix that contains only zeros in the rows which have Dirichlet boundary conditions
        """
        I = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
        I.ident_zeros()
        Mat_bc = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
        [bc.apply(Mat_bc) for bc in self.bcs]
        Mat = I - Mat_bc
        return Mat
    
    def VectorBC_rhs(self): 
        """Vector that contains boundary condition values in the rows which have Dirichlet boundary conditions
        """        
        Vec_bc = assemble(Constant(0.0)*dot(self.func, self.v) * dx)
        [bc.apply(Vec_bc) for bc in self.bcs]
        return Vec_bc
        
class TestBoundaryCondition:
    """
    boundary conditions of cases for testing
    """
    
    def __init__(self):
        self.mesh_list=['cylinder_8k.xml','cylinder_13k.xml','cylinder_26k.xml','cylinder_74k_sym_60ds_40us.xml']
        self.bc_list=['homogeneous','inhomogeneous']
        
    def get_boundaryconditions(self,mesh_name='cylinder_26k.xml',bc_type='homogeneous'):
        if mesh_name in self.mesh_list and bc_type in self.bc_list:
            return eval('self.__'+mesh_name[0:-4]+"('"+bc_type+"')")
        else:
            raise ValueError('boundary conditions type is not in the default list')
    
    def __cylinder_homogeneous(self):
        BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top',     'Mark': 1},
                            'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom',  'Mark': 2},
                            'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet',   'Mark': 3},
                            'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder','Mark': 5},
                            'Outlet'  : {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet',  'Mark': 4}
                            }
        return BoundaryConditions
                   
    def __cylinder_inhomogeneous(self):
        BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top',     'Mark': 1},
                            'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom',  'Mark': 2},
                            'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((1.0,0.0)), 'Boundary': 'inlet',   'Mark': 3},
                            'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder','Mark': 5},
                            'Outlet'  : {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet',  'Mark': 4}
                            }
        return BoundaryConditions
        
    def __cylinder_8k(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')
                   
    def __cylinder_13k(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')
            
    def __cylinder_26k(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')
                   
    def __cylinder_74k_sym_60ds_40us(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')


class TestBoundary:
    """
    boundaries of meshes for testing
    """
    def __init__(self):
        self.mesh_list=['cylinder_8k.xml','cylinder_13k.xml','cylinder_26k.xml','cylinder_74k_sym_60ds_40us.xml']

    def get_boundary(self,mesh_name='cylinder_26k.xml'):
        if mesh_name in self.mesh_list:
            return eval('self.__'+mesh_name[0:-4]+'()')
        else:
            raise ValueError('boundary type is not in the test list')
    
    def __cylinder_8k(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 12.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -12.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 20.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                            }
        return BoundaryLocations
                   
    def __cylinder_13k(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                           }
        return BoundaryLocations
            
    def __cylinder_26k(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                            }
                            
        return BoundaryLocations
                   
    def __cylinder_74k_sym_60ds_40us(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 40.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -40.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and near(x[0], -60.0, tol)'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 60.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                           }
                           
        return BoundaryLocations