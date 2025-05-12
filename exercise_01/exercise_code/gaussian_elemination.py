import numpy as np


def swap_rows(A, i, j):
    """
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the first row
    - j: int, index of the second row

    Outputs:
    - numpy.ndarray, matrix with swapped rows
    """
    A[[i, j]] = A[[j, i]]
    return A


def multiply_row(A, i, scalar):
    """
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row
    - scalar: float, scalar to multiply the row with

    Outputs:
    - numpy.ndarray, matrix with multiplied row
    """
    A[i] = A[i] * scalar
    return A


def add_row(A, i, j, scalar=1):
    """
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row to be added to
    - j: int, index of the row to be added

    Outputs:
    - numpy.ndarray, matrix with added rows
    """
    A[i] = A[i] + A[j] * scalar
    return A


def perform_gaussian_elemination(A):
    """
    Inputs:
    - A: numpy.ndarray, matrix of shape (dim, dim)

    Outputs:
    - ops: List[Tuple[str,int,int]], sequence of elementary operations
    - A_inv: numpy.ndarray, inverse of A
    """
    dim = A.shape[0]
    A_inv = np.eye(dim)
    ops = []
    ########################################################################
    # TODO:                                                                #
    # Implement the Gaussian elemination algorithm.                        #
    # Return the sequence of elementary operations and the inverse matrix. #
    #                                                                      #
    # The sequence of the operations should be in the following format:    #
    # • to swap to rows                                                    #
    #   ("S",<row index>,<row index>)                                      #
    # • to multiply the row with a number                                  #
    #   ("M",<row index>,<number>)                                         #
    # • to add multiple of one row to another row                          #
    #   ("A",<row index i>,<row index j>, <number>)                        #
    # Be aware that the rows are indexed starting with zero.               #
    # Output sufficient number of significant digits for numbers.          #
    # Output integers for indices.                                         #
    #                                                                      #
    # Append to the sequence of operations                                 #
    # • "DEGENERATE" if you have successfully turned the matrix into a     #
    #   form with a zero row.                                              #
    # • "SOLUTION" if you turned the matrix into the $[I|A −1 ]$ form.     #
    #                                                                      #
    # If you found the inverse, output it as a second element,             #
    # otherwise return None as a second element                            #
    ########################################################################

    A_augmented = np.hstack((A, A_inv))
    A_augmented = np.array(A_augmented, dtype=float)

    for i in range(dim):
        # Find the pivot element
        pivot_row = i
        while pivot_row < dim and A_augmented[pivot_row, i] == 0:
            pivot_row += 1

        if pivot_row == dim:
            # matrix degenerated
            break

        if pivot_row != i:
            A_augmented = swap_rows(A_augmented, i, pivot_row)
            ops.append(("S", i, pivot_row))

        # Normalize the pivot row
        pivot_value = A_augmented[i, i]
        if pivot_value != 1:
            A_augmented = multiply_row(A_augmented, i, 1 / pivot_value)
            ops.append(("M", i, 1 / pivot_value))

        # Eliminate the other rows
        for j in range(dim):
            if j != i:
                factor = A_augmented[j, i]
                if factor != 0:
                    A_augmented = add_row(A_augmented, j, i, -factor)
                    ops.append(("A", j, i, -factor))

    # Check if the first part of the augmented matrix is the identity matrix
    if np.allclose(A_augmented[:, :dim], np.eye(dim)):
        ops.append("SOLUTION")
        A_inv = A_augmented[:, dim:]
    else:
        ops.append("DEGENERATE")
        A_inv = None

    return ops, A_inv

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
