import numpy as np

"""This module provides function for error or residual calculations
"""

def rmse(predictions, targets):
    """Root-mean-square deviation between two arrays

    Parameters
    ----------------------------
    predictions : Predicted array

    targets : Obtained array

    Returns
    ----------------------------
    Root-mean-square deviation

    """
    return np.sqrt(((predictions - targets) ** 2).mean())
