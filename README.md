# FERePack

### Flow, Resolvent, and Control

**Resolvent Analysis and Optimal Control Design for Fluid Flows based on Finite Element Method (FEM)**

---

## Overview

FERePack is a comprehensive package for conducting frequency-domain analysis and designing optimal control for fluid dynamics. This package is built on top of the FEniCS project and provides tools for:

- **Frequency Analysis:**
  - Global Singular Value Decomposition (SVD) for Resolvent Analysis.
  - Eigenvalue Solvers for Linear Stability Analysis.
  - Frequency Response for Input-Output Analysis.
  
- **Time-Domain Analysis:**
  - Nonlinear Newton Solvers for stationary flows.
  - Time-stepping solvers for transient flow problems.
  - IPCS (Incremental Pressure Correction Scheme) solvers for time-dependent Navier-Stokes equations.
  
- **Control Design:**
  - Optimal control design based on H2-norm, leveraging libraries like PyMESS or MMESS for solving Riccati equations.

---

## Key Features

- **Resolvent Analysis**: Gain insight into the most responsive modes of the flow to external disturbances.
- **Linear Stability Analysis**: Evaluate the stability of fluid flows by computing the eigenvalues of linearized systems.
- **Frequency Response**: Assess the system's behavior to external forcing in the frequency domain.
- **Optimal Control**: Design controllers to minimize energy or stabilize unstable modes of the fluid system using modern control theory techniques.

---

## Prerequisites

Before using FERePack, ensure that you have installed the following dependencies:

1. **FEniCS** (>= 2019.1.0)  
   [Download FEniCS](https://fenicsproject.org/download/archive/)

2. **MATLAB Engine for Python** (>= 24.1.0)  
   [MATLAB Engine Installation Guide](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

3. **M.E.S.S. Library** (>= 3.0)  
   [M.E.S.S. Documentation](https://www.mpi-magdeburg.mpg.de/projects/mess)

---

## Installation

To install the package and its dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://bitbucket.org/kingboxing/ferepack.git
   cd ferepack
2. Install the package using setup.py:
   ```bash 
   python setup.py install
3. Alternatively, install the dependencies listed in requirements.txt:
   ```bash 
   pip install -r requirements.txt

---

## Usage

Once installed, FERePack can be used for various types of fluid dynamic simulations and analyses. The modules support:

- **Resolvent and Frequency Response Analysis** for determining flow sensitivities.
- **Navier-Stokes Solvers** for both steady-state and transient fluid dynamics.
- **Optimal Control** methods for stabilizing flows or minimizing energy consumption.

Make sure to review the documentation and example scripts for usage guidelines.

---

## Contributing

We welcome contributions! Please follow the standard pull request procedures, ensure your code is well-documented, and include relevant tests.

---

## License

This project is licensed under the terms of the LGPL 2.1 License.

---

## Contact

For further information or to report issues, please contact:

**Bo Jin**  
[jinbo199188@gmail.com](mailto:jinbo199188@gmail.com)

---

## Future Improvements

- Add sparse representation of feedback matrix in eigen_decompose() and relevant solver
- Add solver for Rosenbrock Matrix
- Integration with more analysis and control algorithms.
- Extend to compressible flows.
- Expanded support for high-performance computing (HPC).
