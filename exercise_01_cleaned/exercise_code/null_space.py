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
    U, S, Vt = np.linalg.svd(D, full_matrices=True)

    # Take last (n - rank) rows
    rank = np.sum(S > tolerance)
    null_vector = Vt[rank:]
    #############################################ß###########################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return null_vector
