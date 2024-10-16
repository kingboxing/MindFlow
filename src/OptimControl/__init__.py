# FERePack/src/OptimalControl/__init__.py
"""
Modules related to optimal control, including LQG/LQR solvers, Riccati solvers, and system models.
"""

"""
 The first order DAE2 system

      E_ * z'(t) = A_ * z(t) + B * d(t)
            y(t) = C * u(t)
           
 is encoded in the eqn structure
 The structure of A and E are given as below:

 A_ =   [ A    G;
          G.T  0 ]
 E_ =   [ M    0;
          0    0 ]
 Z(t) =   [ u(t)    p(t) ].T

 G needs to have full column-rank and G.T full row-rank. B and C.T has the same number of rows.

 Solve continuous-time Lyapunov equations with sparse coefficients

   eqn.type = 'N'
     A*X*E' + E*X*A' - E*X*C'*C*X*E' + B*B' = 0
   or
     A*X*E' + E*X*A' - E*X*C'*Q\C*X*E' + B*R*B' = 0

   eqn.type = 'T'
     A'*X*E + E'*X*A - E'*X*B*B'*X*E + C'*C = 0
   or
     A'*X*E + E'*X*A - E'*X*B*R\B'*X*E + C'*Q*C = 0

 Here X = Z*Z' or X = Z*D*Z' with param.LDL_T = True if newton method or RADI method is used. 

 Matrix A can have the form A = Ã + U*V' if U (eqn.U) and V (eqn.V) are
 provided U and V are dense (n x m3) matrices and should satisfy m3 << n

 Solve continuous-time Riccati equations with sparse coefficients with the RADI method, solution can be 
 X = Z*inv(Y)*Z' with param.radi.getZZT = False. If only K0 is provided, only the final gain K will be 
 returned. Otherwise, please provide Z0. 

 The feedback matrix K can be accumulated during the iteration:
     eqn.type = 'N' -> K = (E*X*C')' or K = (E*X*C')'/Q
     eqn.type = 'T' -> K = (E'*X*B)' or K = (E'*X*B)'/R

 In Newton method,
    if param.LDL_T = True then X = Z * D * Z'
    if param.LDL_T = False then X = Z * Z'

 In RADI method, 
    if param.LDL_T = False and param.radi.getZZT = True then X = Z * Z'
    if param.LDL_T = False and param.radi.getZZT = False then X = Z * inv(Y) * Z'
    if param.LDL_T = True then X = Z * D * Z'
 

"""

from .ControlSolver import *
from .LQESolver import *
from .LQRSolver import *
from .RiccatiSolver import *
from .SystemModel import *
from .BernoulliSolver import *

