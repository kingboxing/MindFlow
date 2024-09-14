import numpy as np

def eliminate_duplicates(arr):
    """
    Eliminate duplicates from the array and return the unique array and the indices of the eliminated duplicates.
    
    Parameters:
    arr : array_like
        Input array from which duplicates will be removed.
        
    Returns:
    unique_arr : ndarray
        Array with duplicates removed.
    duplicate_indices : list
        Indices of the eliminated duplicates in the original array.
    """
    # Get unique values and the indices of the first occurrence of each unique value
    unique_arr, indices_first_occurrence = np.unique(arr, return_index=True)
    
    # Find all indices, then compute the difference to get duplicate indices
    all_indices = np.arange(len(arr))
    duplicate_indices = list(set(all_indices) - set(indices_first_occurrence))
    
    return unique_arr, sorted(duplicate_indices)

# Example usage

arr = np.array([5, 3, 8, 3, 9, 1, 5, 8])

unique_arr, duplicate_indices = eliminate_duplicates(arr)

print("Unique array:", unique_arr)
print("Indices of eliminated duplicates:", duplicate_indices)