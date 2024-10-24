# FERePack/src/Params/__init__.py
"""
The `Params` subpackage provides a centralized management system for default parameters used throughout the FERePack package.

It includes the `DefaultParameters` class, which allows for accessing, modifying, and resetting default parameter values for various modules, classes, or functions. This facilitates consistent parameter configurations and enhances flexibility in controlling solver and analysis settings.

Modules and Classes
-------------------

- **Params**:
    - Contains the `DefaultParameters` class for managing default parameters.
    - **Classes**:
        - `DefaultParameters`: Manages default parameters for different modules, classes, or functions.

Usage
-----

To utilize the default parameter management system provided by the `Params` subpackage, you can import the `DefaultParameters` class as follows:

```python
from FERePack.Params import DefaultParameters
```

Notes
-----
- Customization: You can access and modify default parameters for specific modules or classes, enabling flexibility in configuring solvers and analyses.
- Consistency: Using the DefaultParameters class helps maintain consistent parameter settings across different components of the FERePack package.

Examples
-----
Example of accessing and updating default parameters:

```python
from FERePack.Params import DefaultParameters

# Initialize the default parameters manager
params_manager = DefaultParameters()

# Get default parameters for a specific module
eigen_solver_defaults = params_manager.get_defaults('eigen_solver')

# Update default parameters for a module
new_defaults = {'solver_type': 'custom_solver', 'custom_param': 42}
params_manager.update_defaults('eigen_solver', new_defaults)

# Reset default parameters for a module
params_manager.reset_defaults('eigen_solver')
```
In this example:

- We import the DefaultParameters class from FERePack.Params.
- Initialize an instance of DefaultParameters.
- Access default parameters for the 'eigen_solver' module.
- Update the default parameters for 'eigen_solver' with new values.
- Reset the default parameters for 'eigen_solver' to their initial settings.

"""

from .Params import DefaultParameters