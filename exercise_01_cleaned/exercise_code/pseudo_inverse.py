import numpy as np


def solve_linear_equation_SVD(D, b):
    """
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    - b: numpy.ndarray, vector of shape (m,)
    Outputs:
    - x: numpy.ndarray, solution of the linear equation D*x = b
    - D_inv: numpy.ndarray, pseudo-inverse of D of shape (n,m)
    """
    x, D_inv = None, None
    ########################################################################
    # TODO:                                                                #
    # Solve the linear equation D*x = b using the pseudo-inverse and SVD.  #
    # Your code should be able to tackle the case where D is singular.     #
    ########################################################################

    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    tolerance = 1e-10 * np.max(S)
    S_inv = np.array([1 / s if s > tolerance else 0 for s in S])

    # Compute the pseudo-inverse of D
    D_inv = Vt.T @ np.diag(S_inv) @ U.T

    # Solve: D*x = b -> x = D_inv @ b
    x = D_inv @ b

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return x, D_inv
