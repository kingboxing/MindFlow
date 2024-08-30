#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:48:11 2024

@author: bojin
"""
from src.Deps import *

def allclose_spmat(A, B, tol=1e-12):
    """
    Check if two sparse matrices A and B are identical within a tolerance.

    Parameters:
    - A, B: scipy.sparse matrices
    - tol: tolerance value

    Returns:
    - True if matrices are identical within the specified tolerance, otherwise False.
    """
    if A.shape != B.shape:
        return False

    # Compute the difference between the matrices
    diff = A - B

    # Compute the norm of the difference
    diff_norm = spla.norm(diff, ord='fro')

    # Check if the norm is within the tolerance
    return diff_norm <= tol


def get_subspace_info(function_space):
    """
    Get the number of top-level sub-elements and the total number of scalar subspaces 
    in a given two-level function space.

    Parameters
    ----------
    function_space : FunctionSpace
        DESCRIPTION.

    Returns
    -------
    sub_spaces : tuple
        The number of scalar subspaces under each subspace.

    """
    sub_spaces = ()
    num_sub_spaces = function_space.num_sub_spaces()
    for i in range(num_sub_spaces):
        num_space = function_space.sub(i).num_sub_spaces()
        if num_space == 0:
            sub_spaces += (1, )
            #total_scalar_subspaces += 1
        else:
            sub_spaces += (num_space, )
            #total_scalar_subspaces += num_space
    
    return sub_spaces

def find_subspace_index(index, sub_spaces):
    """
    Find the top-level subspace and sub-level subspace indices for a given scalar subspace index.

    Parameters
    ----------
    index : int
        The index of the scalar subspace (0-based index)..
    sub_spaces : tuple
        number scalar subspaces per top level subspace. 
        A list where each entry corresponds to the number of scalar subspaces in a top-level subspace.

    Raises
    ------
    ValueError
        Index out of bounds.

    Returns
    -------
    (top_level_index, sub_level_index) : tuple
        A tuple with the top-level subspace index and the sub-level subspace index (both 0-based)..

    """
    cumulative_sum = 0
    
    for i, num_scalar_subspaces in enumerate(sub_spaces):
        previous_sum = cumulative_sum
        cumulative_sum += num_scalar_subspaces
        if index < cumulative_sum:
            top_level_index = i
            sub_level_index = index - previous_sum
            if sub_spaces[top_level_index]>1:
                return (top_level_index, sub_level_index)
            else: # the number of scalar subspaces in a top-level subspace = 1
                return (top_level_index, )
                
    raise ValueError("Index out of bounds")