# FERePack/__init__.py
"""
FERePack: A package for solving complex frequency-based analysis and control problems using FEniCS.
"""

# Import modules from src/ to make them accessible without 'src.' in the import path
from .src.BasicFunc import *
from .src.Eqns import *
from .src.FreqAnalys import *
from .src.LinAlg import *
from .src.NSolver import *
from .src.OptimControl import *
from .src.Deps import *
