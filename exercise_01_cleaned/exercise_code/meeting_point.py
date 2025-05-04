import numpy as np


def meeting_point_linear(pts_list):
    """
    Inputs:
    - pts_list: List[numpy.ndarray], list of each persons points in the space
    Outputs:
    - numpy.ndarray, meeting point or vectors spanning the possible meeting points of shape (m, dim_intersection)
    """
    A = pts_list[0]  # person A's points of shape (m,num_pts_A)
    B = pts_list[1]  # person B's points of shape (m,num_pts_B)

    ########################################################################
    # TODO:                                                                #
    # Implement the meeting point algorithm.                               #
    #                                                                      #
    # As an input, you receive                                             #
    # - for each person, you receive a list of landmarks in their subspace.#
    #   It is guaranteed that the landmarks span each person’s whole       #
    #   subspace.                                                          #
    #                                                                      #
    # As an output,                                                        #
    # - If such a point exist, output it.                                  #
    # - If there is more than one such point,                              #
    #   output vectors spanning the space.                                 #
    ########################################################################
    tolerance = 1e-10
    m = A.shape[0]

    def null_space(A, tol=tolerance):
        """Compute the null space of A using SVD."""
        U, S, Vh = np.linalg.svd(A)
        rank = (S > tol).sum()
        null_space = Vh[rank:].T
        return null_space

    # SVD to get orthonormal basis for each subspace
    # Output of SVD is U, S, V^T
    U_a, S_a, Vh_a = np.linalg.svd(A, full_matrices=False)
    U_b, S_b, Vh_b = np.linalg.svd(B, full_matrices=False)

    rank_a = np.sum(S_a > tolerance)
    rank_b = np.sum(S_b > tolerance)

    # generate basis for each subspace
    basis_a = U_a[:, :rank_a]
    basis_b = U_b[:, :rank_b]

    # check if the subspaces intersect
    # Build the matrix [basis_a  -basis_b]
    M = np.hstack((basis_a, -basis_b))
    null_space_M = null_space(M)
    null_space_dim = null_space_M.shape[1]

    if null_space_dim == 0:
        print("No meeting point")
        return np.zeros((1, m))

    # Compute the meeting points
    alpha = null_space_M[:rank_a, :]
    meeting_pts = basis_a @ alpha

    # Normalize meeting points (avoid scaling ambiguity)
    norms = np.linalg.norm(meeting_pts, axis=0, keepdims=True)
    meeting_pts_normalized = meeting_pts / (norms + tolerance)

    if null_space_dim == 1:
        print("Unique meeting point")
    else:
        print("Multiple meeting points")
    return meeting_pts_normalized

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
