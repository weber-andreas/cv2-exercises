import numpy as np


def getHarrisCorners(M, kappa, theta):
    # Compute Harris corners
    # Input:
    # M: structure tensor of shape (H, W, 2, 2)
    # kappa: float (parameter for Harris corner score)
    # theta: float (threshold for corner detection)
    # Output:
    # score: numpy.ndarray (Harris corner score) of shape (H, W)
    # points: numpy.ndarray (detected corners) of shape (N, 2)

    ########################################################################
    # TODO:                                                                #
    # Compute the Harris corner score and find the corners.               #
    #                                                                      #
    # Hints:                                                               #
    # - The Harris corner score is computed using the determinant and      #
    #   trace of the structure tensor.                                     #
    # - Use the threshold theta to find the corners.                       #
    # - Use non-maximum suppression to find the corners.                   #
    ########################################################################

    point_list = []
    score = np.zeros((M.shape[0], M.shape[1]))
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):

            s = np.linalg.det(M[x, y]) - kappa * np.power(np.trace(M[x, y]), 2)
            score[x, y] = s
            if s > theta:
                point_list.append([x, y])

    points = np.array(point_list)
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return score, points
