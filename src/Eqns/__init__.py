# FERePack/src/Eqns/__init__.py

"""
The `Eqns` module provides classes and functions for defining and working with equations relevant to computational 
fluid dynamics (CFD), such as the Navier-Stokes equations. It is designed to facilitate the setup and solution of 
fluid flow problems using finite element methods in FEniCS.

Modules and Classes
-------------------

- **NavierStokes**: - `Incompressible`: Class for defining various forms of the incompressible Navier-Stokes
equations, suitable for steady/transient and linear/nonlinear simulations.

Usage
-----

To utilize the `Eqns` module in your simulation workflow, you can import the necessary classes and set up your equations as follows:

```python
from FERePack.Eqns import Incompressible
from FERePack.BasicFunc import TaylorHood, SetBoundary
from dolfin import Mesh

# Define the mesh and finite element
mesh = Mesh("mesh.xml")
element = TaylorHood(mesh)

# Define boundary conditions
boundary = SetBoundary(mesh, element)

# Initialize the incompressible Navier-Stokes equations with a given Reynolds number
ns_eqn = Incompressible(element, boundary, Re=100)

# Define the steady nonlinear Navier-Stokes weak form
NS_form = ns_eqn.SteadyNonlinear()
```

Notes
-----

- **Dependencies**: Ensure that the Deps module and FEniCS are correctly installed and configured in your environment.
- **Finite Elements**: The Incompressible class works with finite element objects defined in the BasicFunc module (e.g., TaylorHood element).
- **Boundary Conditions**: Boundary measures and normals are obtained from the SetBoundary object, which must be properly initialized.

Documentation
-----

Detailed documentation for the Incompressible class and its methods can be found in the NavierStokes.py module. The class provides methods for:

- Defining steady and transient forms of the Navier-Stokes equations.
- Implementing linearized and nonlinear formulations.
- Handling frequency-domain and quasi-steady analyses.
- Setting up source terms and boundary traction expressions.
- Computing auxiliary quantities like force expressions and vorticity.

By providing these tools, the Eqns module allows users to set up complex fluid dynamics problems in a structured and efficient manner.
"""

from .NavierStokes import Incompressible
