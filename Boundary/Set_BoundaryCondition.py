from __future__ import print_function
from fenics import *

"""This module provides classes for setting up boundary conditions
"""


class BoundaryCondition:
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
    >>> from RAPACK.Boundary.Set_Boundary import Boundary
    >>> from RAPACK.Boundary.Set_BoundaryCondition import BoundaryCondition
    >>> from fenics import *
    >>> mesh = Mesh("mesh.xml")
    >>> P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    >>> P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    >>> TH = MixedElement([P2, P1])
    >>> V = FunctionSpace(mesh, TH)
    >>> boundarylocation = 'on_boundary and near(x[0], 0.0, tol)'
    >>> boundarycondition = {'FunctionSpace': 'V.sub(0).sub(0)', 'Value': Constant(0.0)}
    >>> boundary=Boundary(mesh)
    >>> boundary.set_boundary(location = boundarylocations, mark = 1)
    >>> bc = BoundaryCondition(Functionspace=V, boundary=boundary)
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
        
class CommonBoundaryConditions:
    def __init__(self):
        self.mesh_list=['cylinder_8thousand.xml','cylinder_13thousand.xml','cylinder_26thousand.xml','cylinder_74thousand_sym_60downstream_40upstream.xml']
        self.bc_list=['homogeneous','inhomogeneous']
        
    def get_boundaryconditions(self,mesh_name='cylinder_26thousand.xml',bc_type='homogeneous'):
        if mesh_name in self.mesh_list and bc_type in self.bc_list:
            return eval('self.__'+mesh_name[0:-4]+"('"+bc_type+"')")
        else:
            raise ValueError('boundary conditions type is not in the common list')
    
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
        
    def __cylinder_8thousand(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')
                   
    def __cylinder_13thousand(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')
            
    def __cylinder_26thousand(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')
                   
    def __cylinder_74thousand_sym_60downstream_40upstream(self,bc_type):
        return eval('self.__cylinder_'+bc_type+'()')