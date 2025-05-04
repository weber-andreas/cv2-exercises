import numpy as np


def get_null_vector(D):
    """
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    Outputs:
    - null_vector: numpy.ndarray, matrix of shape (dim_kern,n)
    """

    ########################################################################
    # TODO:                                                                #
    # Get the kernel of the matrix D.                                      #
    # the kernel should consider the numerical errors.                     #
    ########################################################################

    # It should hold: D*x=0
    # Use SVD to compute singular vlues in S that are close to 0.
    tolerance = 1e-10
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    # Select rows, columns in V, close to 0
    null_mask = S <= tolerance
    null_vector = Vt[null_mask, :]

    # If the null space is empty, return an empty array
    if null_vector.size == 0:
        null_vector = np.zeros((1, D.shape[1]))

    #############################################ß###########################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return null_vector
