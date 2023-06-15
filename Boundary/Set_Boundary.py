from __future__ import print_function
from fenics import *

"""This module provides subclasses of FEniCS interface for
defining and marking boundaries 
"""


class Boundary(SubDomain):
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
    >>> from RAPACK.Boundary.Set_Boundary import Boundary
    >>> from fenics import *
    >>> mesh = Mesh("mesh.xml")
    >>> BoundaryLocation = 'on_boundary and near(x[0], 0.0, tol)'
    >>> boundary=Boundary(mesh)
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

class CommonBoundary:
    def __init__(self):
        self.mesh_list=['cylinder_8thousand.xml','cylinder_13thousand.xml','cylinder_26thousand.xml','cylinder_74thousand_sym_60downstream_40upstream.xml']

    def get_boundary(self,mesh_name='cylinder_26thousand.xml'):
        if mesh_name in self.mesh_list:
            return eval('self.__'+mesh_name[0:-4]+'()')
        else:
            raise ValueError('boundary type is not in the common list')
    
    def __cylinder_8thousand(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 12.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -12.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 20.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                            }
        return BoundaryLocations
                   
    def __cylinder_13thousand(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                           }
        return BoundaryLocations
            
    def __cylinder_26thousand(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                            }
                            
        return BoundaryLocations
                   
    def __cylinder_74thousand_sym_60downstream_40upstream(self):
        BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 40.0, tol)'},
                           'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -40.0, tol)'},
                           'Inlet'      : {'Mark': 3, 'Location':'on_boundary and near(x[0], -60.0, tol)'},
                           'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 60.0, tol)'},
                           'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                           }
                           
        return BoundaryLocations