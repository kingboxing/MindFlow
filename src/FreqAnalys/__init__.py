# FERePack/src/FreqAnlys/__init__.py
"""
The `FreqAnalys` subpackage provides tools and solvers for frequency domain analysis of fluid flow problems, particularly for linearized incompressible Navier-Stokes equations using the finite element method with the FEniCS library.

It includes base classes and implementations for performing frequency response analysis, stability analysis (eigenvalue analysis), resolvent analysis, and control design by linearizing the Navier-Stokes equations around a base flow and solving the resulting linear system in the frequency domain.

Modules and Classes
-------------------

- **FreqSolverBase**:
    - Contains the base class `FrequencySolverBase` which provides common functionalities for frequency domain solvers.
    - **Classes**:
        - `FrequencySolverBase`: Base class for frequency domain solvers for linearized Navier-Stokes equations.

- **FreqSolver**:
    - Implements specific solvers for frequency response analysis.
    - **Classes**:
        - `FrequencyResponse`: Solves the frequency response of a linearized Navier-Stokes system with specified input and output vectors.

- **EigenSolver**:
    - Implements solvers for eigenvalue and eigenvector analysis.
    - **Classes**:
        - `EigenAnalysis`: Performs eigenvalue and eigenvector analysis of the linearized Navier-Stokes system.

- **ResolSolver**:
    - Implements solvers for resolvent analysis.
    - **Classes**:
        - `ResolventAnalysis`: Performs resolvent analysis to compute the most amplified modes of the system under harmonic forcing.

Usage
-----

To utilize the tools provided by the `FreqAnalys` subpackage, you can import the necessary classes as follows:

```python
from FERePack.FreqAnalys import FrequencySolverBase, FrequencyResponse, EigenAnalysis, ResolventAnalysis
```
Notes
-----
- **Dependencies**: Ensure that FEniCS and other required dependencies are installed and properly configured.
- **Finite Element Spaces**: The solvers use finite element spaces appropriate for fluid dynamics problems, such as Taylor-Hood elements.
- **Customization**: The base class can be subclassed to implement specific analyses such as computing the frequency response, performing eigenvalue analysis, resolvent analysis, or designing controllers.

Examples
-----

Example of performing frequency response analysis using FrequencyResponse:

```python
from FERePack.FreqAnalys import FrequencyResponse
import numpy as np

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0
frequency = 1.0  # Frequency in Hz
omega = 2 * np.pi * frequency  # Angular frequency

# Initialize the frequency response solver
solver = FrequencyResponse(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set base flow
solver.set_baseflow(ic=base_flow_function)

# Define input and output vectors
input_vec = ...  # Define your input vector (actuation)
output_vec = ...  # Define your output vector (measurement)

# Solve for the frequency response at the specified frequency
s = 1j * omega
solver.solve(s=s, input_vec=input_vec, output_vec=output_vec)

# Access the computed gain (frequency response)
gain = solver.gain
```

Example of performing eigenvalue analysis using EigenAnalysis:

```python
from FERePack.FreqAnalys import EigenAnalysis

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0

# Initialize the eigenvalue analysis solver
solver = EigenAnalysis(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set base flow
solver.set_baseflow(ic=base_flow_function)

# Solve for the leading eigenvalues and eigenvectors
solver.solve(k=5, sigma=0.0)

# Access the computed eigenvalues and eigenvectors
eigenvalues = solver.vals
eigenvectors = solver.vecs
```
Example of performing resolvent analysis using ResolventAnalysis:

```python
from FERePack.FreqAnalys import ResolventAnalysis
import numpy as np

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0
frequency = 1.0  # Frequency in Hz
omega = 2 * np.pi * frequency  # Angular frequency

# Initialize the resolvent analysis solver
solver = ResolventAnalysis(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set base flow
solver.set_baseflow(ic=base_flow_function)

# Solve for the leading singular modes
s = 1j * omega
solver.solve(k=5, s=s)

# Access the computed singular values and modes
singular_values = solver.energy_amp
force_modes = solver.force_mode
response_modes = solver.response_mode
```

In these examples:

- We import the necessary solver classes from FERePack.FreqAnalys.
- Set up the computational mesh and physical parameters.
- Define boundary conditions and base flow.
- Use the appropriate solver (FrequencyResponse, EigenAnalysis, or ResolventAnalysis) to perform the analysis.
- Access the results from the solver’s attributes.
"""
from .EigenSolver import *
from .FreqSolver import *
from .ResolSolver import *
from .FreqSolverBase import *

