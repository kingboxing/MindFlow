# FERePack/src/OptimControl/__init__.py
"""
The `OptimControl` subpackage provides modules related to optimal control, including Linear Quadratic Gaussian (
LQG), Linear Quadratic Regulator (LQR) solvers, Riccati solvers, and system models tailored for fluid dynamics
applications.

It contains classes and functions for constructing and analyzing control systems for fluid flows, particularly those
represented as Differential-Algebraic Equations of Index 2 (DAE2). The subpackage leverages advanced numerical
methods and the M.E.S.S. (Matrix Equation Sparse Solver) library to solve generalized Riccati equations and design
optimal controllers and estimators.

Overview
--------

The first-order DAE2 system is given by:

```
    E_ * z'(t) = A_ * z(t) + B_ * d(t)
          y(t) = C_ * z(t)
```

where:

- `E_` and `A_` are the system matrices structured as:

```
      E_ = [ M    0 ]				A_ = [ A    G ]
           [ 0    0 ]					 [ G^T  0 ]

      B_ = [ B ]					C_ = [ C ]
           [ 0 ]						 [ 0 ]
```

- `M` is the mass matrix.
- `A` is the system matrix.
- `G` needs to have full column rank, and `G^T` needs to have full row rank. They represent the gradient and divergence operators, respectively.
- `B` is the input (actuation) matrix.
- `C` is the output (measurement) matrix.
- `z(t)` is the state vector, partitioned as `z(t) = [ u(t); p(t) ]`, where `u(t)` is the velocity field and `p(t)` is the pressure field.
- `d(t)` is the actuation signal.

The solvers included in this subpackage can handle the following equations:

- **Continuous Riccati equations with sparse coefficients:**
	(ignore _ notation)

  - For `eqn.type = 'N'` (standard):

  	```
	A * X * E' + E * X * A' - E * X * C' * C * X * E' + B * B' = 0


  or


   A * X * E' + E * X * A' - E * X * C' * Q⁻¹ * C * X * E' + B * R * B' = 0
   ```
  when param.LDL_T = True.

  - For `eqn.type = 'T'` (transposed):

  ```
  A' * X * E + E' * X * A - E' * X * B * B' * X * E + C' * C = 0


  or


  A' * X * E + E' * X * A - E' * X * B * R⁻¹ * B' * X * E + C' * Q * C = 0
  ```
  when param.LDL_T = True.

- **Solution of Continuous Riccati equations:**

  Solutions can be obtained in the form:

  - `X = Z * Z'` or `X = Z * D * Z'` when `param.LDL_T = True`.
  - `X = Z * inv(Y) * Z'` when `param.radi.getZZT = False` and `param.LDL_T = False`.

  The feedback matrix `K` can be accumulated during the iteration:

  - For `eqn.type = 'N'`:

	```
    K = (E * X * C')'    or    K = (E * X * C')' * Q⁻¹
    ```

  - For `eqn.type = 'T'`:

	```
    K = (E' * X * B)'    or    K = (E' * X * B)' * R⁻¹
    ```

- **Matrix A with feedback form:**

  Matrix `A` can have the form `A = Ã + U * V'` if `U` (`eqn.U`) and `V` (`eqn.V`) are provided.
  `U` and `V` are dense matrices of size `(n x m3)` and should satisfy `m3 << n`.

Modules and Classes
-------------------

- **SystemModel**:
    - Contains the `StateSpaceDAE2` class for assembling the state-space model of the linearized Navier-Stokes equations (DAE2 type).
    - **Classes**:
        - `StateSpaceDAE2`: Assembles the state-space model suitable for control design and analysis.

- **BernoulliSolver**:
    - Implements the `BernoulliFeedback` class for computing feedback control using a generalized algebraic Bernoulli equation.
    - **Classes**:
        - `BernoulliFeedback`: Computes feedback control to stabilize a dynamical system by solving a generalized Bernoulli equation.

- **RiccatiSolver**:
    - Implements the `GRiccatiDAE2Solver` class for solving generalized Riccati equations for index-2 systems.
    - **Classes**:
        - `GRiccatiDAE2Solver`: Solves generalized Riccati equations and computes performance metrics like the H2 norm.

- **LQESolver**:
    - Implements the `LQESolver` class for solving the Linear Quadratic Estimation (LQE) problem for index-2 systems.
    - **Classes**:
        - `LQESolver`: Solves the LQE problem and computes the estimator (Kalman filter) gain.

- **LQRSolver**:
    - Implements the `LQRSolver` class for solving the Linear Quadratic Regulator (LQR) problem for index-2 systems.
    - **Classes**:
        - `LQRSolver`: Solves the LQR problem and computes the optimal control gain.

Usage
-----

To utilize the tools provided by the `OptimControl` subpackage, you can import the necessary classes as follows:

```python
from FERePack.OptimControl import StateSpaceDAE2, BernoulliFeedback, GRiccatiDAE2Solver, LQESolver, LQRSolver
```

Notes
-----

- Dependencies: Ensure that FEniCS, the M.E.S.S. library, and other required dependencies are installed and properly configured.
- Control Design: The subpackage is intended for use in advanced control design applications, requiring familiarity with control theory and numerical methods.
- Integration: Designed to integrate seamlessly with other FERePack subpackages, such as FreqAnalys and Params.

Examples
----------

Example of assembling a state-space model and solving the LQG problem using LQESolver and LQRSolver:

```python
from FERePack.OptimControl import StateSpaceDAE2, LQESolver, LQRSolver

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0

# Initialize the state-space model
model = StateSpaceDAE2(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
model.set_boundary(bc_list)
model.set_boundarycondition(bc_list)

# Set base flow
model.set_baseflow(ic=base_flow_function)

# Define input and output vectors
input_vec = ...   # Define your input vector (actuation)
output_vec = ...  # Define your output vector (measurement)

# Assemble the state-space model
model.assemble_model(input_vec=input_vec, output_vec=output_vec)

# Initialize the LQE solver with the model
lqe_solver = LQESolver(model, method='nm', backend='python')

# Set sensor noise parameters
lqe_solver.sensor_noise(alpha=1.0)

# Set disturbance parameters
disturbance_matrix = ...  # Define your disturbance matrix
lqe_solver.disturbance(B=disturbance_matrix, beta=0.1)

# Solve the LQE problem
lqe_solver.solve()

# Compute the estimator (Kalman filter) gain
estimator_gain = lqe_solver.estimator()

# Initialize the LQR solver with the model
lqr_solver = LQRSolver(model, method='nm', backend='python')

# Set control penalty parameters
lqr_solver.control_penalty(alpha=1.0)

# Set measurement parameters
measurement_matrix = ...  # Define your measurement matrix
lqr_solver.measurement(C=measurement_matrix, beta=0.1)

# Solve the LQR problem
lqr_solver.solve()

# Compute the regulator (optimal control) gain
regulator_gain = lqr_solver.regulator()
```

In this example:

- We import the necessary classes from FERePack.OptimControl.
- Initialize an instance of StateSpaceDAE2 with the mesh and physical parameters.
- Set boundary conditions and base flow.
- Define the input and output vectors representing the actuation and measurement of the system.
- Assemble the state-space model using the assemble_model method.
- Use LQESolver to solve the LQE problem and compute the estimator gain.
- Use LQRSolver to solve the LQR problem and compute the optimal control gain.

Notes on Equations and Solvers
--------------------------------

- **Feedback Matrix K Accumulation**:

	During iterative solving, the feedback matrix K can be accumulated based on the equation type:

	- For eqn.type = 'N':

	```
	K = (E * X * C')'    or    K = (E * X * C')' * Q⁻¹
	```

	- For eqn.type = 'T':

	```
	K = (E' * X * B)'    or    K = (E' * X * B)' * R⁻¹
	```

- **Solution Forms**:
    - In Newton’s method:
	    - If param.LDL_T = True, then X = Z * D * Z'.
        - If param.LDL_T = False, then X = Z * Z'.
    - In RADI method:
        - If param.LDL_T = False and param.radi.getZZT = True, then X = Z * Z'.
        - If param.LDL_T = False and param.radi.getZZT = False, then X = Z * inv(Y) * Z'.
        - If param.LDL_T = True, then X = Z * D * Z'.
"""

from .ControlSolver import *
from .LQESolver import *
from .LQRSolver import *
from .RiccatiSolver import *
from .SystemModel import *
from .BernoulliSolver import *
