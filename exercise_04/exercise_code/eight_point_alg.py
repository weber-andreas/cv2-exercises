import numpy as np

from .utils import skewMat


def transformImgCoord(x1, x2, y1, y2, K1, K2):
    # transform the image coordinates
    # assume the image plane is at z = 1
    # what should 3D points be in camera coordinates?
    # input: 2D points in two images (x1, x2, y1, y2), intrinsics K1, K2
    # output: normalized camera coords x1, x2, y1, y2 (each of shape (n_pts,))

    ########################################################################
    # TODO: Implement the transformation with                              #
    # the given camera intrinsic matrices                                  #
    ########################################################################
    # Z = 1
    # x1 = (x1 - K1[0, 2]) * Z / K1[0, 0]
    # y1 = (y1 - K1[1, 2]) * Z / K1[1, 1]

    # x2 = (x2 - K2[0, 2]) * Z / K2[0, 0]
    # y2 = (y2 - K2[1, 2]) * Z / K2[1, 1]

    # Convert pixel coords to homogeneous
    pts1 = np.vstack((x1, y1, np.ones_like(x1)))  # shape: (3, N)
    pts2 = np.vstack((x2, y2, np.ones_like(x2)))  # shape: (3, N)

    # Normalize using K^{-1}
    pts1_norm = np.linalg.inv(K1) @ pts1  # shape: (3, N)
    pts2_norm = np.linalg.inv(K2) @ pts2  # shape: (3, N)

    # Extract x, y components
    x1, y1 = pts1_norm[0], pts1_norm[1]
    x2, y2 = pts2_norm[0], pts2_norm[1]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return x1, x2, y1, y2


def constructChiMatrix(x1, x2, y1, y2):
    # construct the chi matrix using the kronecker product
    # input: normalized camera coords x1, y1 in image1 and x2, y2 in image2
    # output: chi matrix of shape (n_pts, 9)
    n_pts = x1.shape[0]
    chi_mat = np.zeros((n_pts, 9))
    for i in range(n_pts):
        ########################################################################
        # TODO: construct the chi matrix by kronecker product                  #
        ########################################################################
        a = np.kron(
            np.array([[x1[i], y1[i], 1]]), np.array([[x2[i]], [y2[i]], [1]])
        ).flatten()
        chi_mat[i, :] = a

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    return chi_mat


def solveForEssentialMatrix(chi_mat):
    # project the essential matrix onto the essential space
    # input: chi matrix - shape (n_pts, 9)
    # output: essential matrix E - shape (3, 3), U, Vt - shape (3, 3),  S - shape (3, 3) diagonal matrix with E = U @ S @ Vt

    ########################################################################
    # TODO: solve the minimization problem to get the solution of E here.  #
    ########################################################################

    # Compute the null space of chi matrix, last column in V
    _, _, Vt = np.linalg.svd(chi_mat)
    E_vec = Vt[-1]  # Last row in Vt is null vector
    # E_vec = E_vec / np.linalg.norm(E_vec)  # Normalize the vector
    E = E_vec.reshape(3, 3)

    # Enforce essential matrix constraints via SVD projection
    U, S_raw, Vt = np.linalg.svd(E)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    # ensure the determinant of U and Vt is positive
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    ########################################################################
    # TODO: Project the E to the normalized essential space here,          #
    # don't forget S should be a diagonal matrix.                          #
    ########################################################################
    # sigma = (S_raw[0] + S_raw[1]) / 2
    S = np.diag([1, 1, 0])
    E = U @ S @ Vt

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return E, U, Vt, S


def constructEssentialMatrix(x1, x2, y1, y2, K1, K2):
    # compute an approximate essential matrix
    # input: 2D points in two images (x1, x2, y1, y2), camera intrinsic matrix K1, K2
    # output: essential matrix E - shape (3, 3),
    #         singular vectors of E: U, Vt - shape (3, 3),
    #         singular values of E: S - shape (3, 3) diagonal matrix, with E = U @ S @ Vt.

    # you need to finish the following three functions
    x1, x2, y1, y2 = transformImgCoord(x1, x2, y1, y2, K1, K2)
    chi_mat = constructChiMatrix(x1, x2, y1, y2)
    E, U, Vt, S = solveForEssentialMatrix(chi_mat)
    return E, U, Vt, S


def Rz(theta_rad):
    """
    Generates a 3x3 rotation matrix around the Z-axis.

    Args:
        theta_rad (float): Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 Z-axis rotation matrix.
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def recoverPose(U, Vt, S):
    # recover the possible poses from the essential matrix
    # input: singular vectors of E: U, Vt - shape (3, 3),
    #        singular values of E: S - shape (3, 3) diagonal matrix, with E = U @ S @ Vt.
    # output: possible rotation matrices R1, R2 - each of shape (3, 3),
    #         possible translation vectors T1, T2 - each of shape (3,)

    ########################################################################
    # TODO: 1. implement the R_z rotation matrix.                          #
    #          There should be two of them.                                #
    #       2. recover the rotation matrix R                               #
    #          with R_z, U, Vt. (two of them).                             #
    #       3. recover \hat{T} with R_z, U, S                              #
    #          and extract T. (two of them).                               #
    #       4. return R1, R2, T1, T2.                                      #
    ########################################################################
    assert np.allclose(
        Rz(np.pi / 2), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), atol=1e-6
    ), "Rz(np.pi / 2) is not correct"

    R1 = U @ Rz(np.pi / 2).T @ Vt
    R2 = U @ Rz(-np.pi / 2).T @ Vt

    T1_skew = U @ Rz(np.pi / 2).T @ S @ U.T
    T2_skew = U @ Rz(-np.pi / 2).T @ S @ U.T

    T1 = np.array([-T1_skew[1, 2], T1_skew[0, 2], -T1_skew[0, 1]])
    T2 = np.array([-T2_skew[1, 2], T2_skew[0, 2], -T2_skew[0, 1]])

    # Ensure T1 and T2 are unit vectors
    T1 /= np.linalg.norm(T1)
    T2 /= np.linalg.norm(T2)

    # Ensure R1 and R2 are valid rotation matrices
    assert np.isclose(np.linalg.det(R1), 1), "R1 is not a valid rotation matrix"
    assert np.isclose(np.linalg.det(R2), 1), "R2 is not a valid rotation matrix"

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return R1, R2, T1, T2


def reconstruct(x1, x2, y1, y2, R, T):
    # reconstruct the 3D points from the 2D correspondences and (R, T)
    # input:  normalized camera coords in two images (x1, x2, y1, y2), rotation matrix R - shape (3, 3), translation vector T - shape (3,)
    # output: 3D points X1, X2

    n_pts = x1.shape[0]
    X1, X2 = None, None

    ########################################################################
    # TODO: implement the structure reconstruction matrix M.               #
    #  1. construct the matrix M -shape (3 * n_pts, n_pts + 1)             #
    #    which is defined as page18, chapter 5.                            #
    #  2. find the lambda and gamma as explained on the same page.         #
    #     make sure that gamma is positive                                 #
    #  3. generate the 3D points X1, X2 with lambda and (R, T).            #
    #  4. check the number of points with positive depth,                  #
    #     it should be n_pts                                               #
    ########################################################################

    M = np.zeros((3 * n_pts, n_pts + 1))
    for i in range(n_pts):
        x1_h = np.array([x1[i], y1[i], 1.0])
        x2_h = np.array([x2[i], y2[i], 1.0])

        skew_x2 = skewMat(x2_h)

        M[3 * i : 3 * i + 3, i] = skew_x2 @ R @ x1_h
        M[3 * i : 3 * i + 3, -1] = skew_x2 @ T

    # Solve for lambdas and gamma
    _, _, Vh = np.linalg.svd(M)
    lambda_gamma = Vh[-1]  # last row of V^T corresponds to smallest singular value

    lambdas = lambda_gamma[:-1]
    gamma = lambda_gamma[-1]

    if gamma < 0:
        lambdas = -lambdas
        gamma = -gamma

    # Fix scale ambiguity
    lambdas = lambdas / gamma

    # Reconstruct 3D points in both camera frames
    homog_x1 = np.vstack([x1, y1, np.ones(n_pts)])

    X1 = lambdas * homog_x1
    X2 = R @ X1 + T[:, np.newaxis]

    # Count how many points have positive depth (Z > 0)
    n_positive_depth1 = np.sum(X1[2, :] > 0)
    n_positive_depth2 = np.sum(X2[2, :] > 0)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    if n_positive_depth1 == n_pts and n_positive_depth2 == n_pts:
        return X1, X2
    else:
        return None, None


def allReconstruction(x1, x2, y1, y2, R1, R2, T1, T2, K1, K2):
    # reconstruct the 3D points from the 2D correspondences and the possible poses
    # input: 2D points in two images (x1, x2, y1, y2), possible rotation matrices R1, R2 - each of shape (3, 3),
    #        possible translation vectors T1, T2 - each of shape (3,), intrinsics K1, K2
    # output: the correct rotation matrix R, translation vector T, 3D points X1, X2

    num_sol = 0
    # transform to camera coordinates
    x1, x2, y1, y2 = transformImgCoord(x1, x2, y1, y2, K1, K2)
    # first check (R1, T1)
    X1, X2 = reconstruct(x1, x2, y1, y2, R1, T1)
    if X1 is not None:
        num_sol += 1
        R = R1
        T = T1
        X1_res = X1
        X2_res = X2

    # check (R1, T2)
    X1, X2 = reconstruct(x1, x2, y1, y2, R1, T2)
    if X1 is not None:
        num_sol += 1
        R = R1
        T = T2
        X1_res = X1
        X2_res = X2

    # check (R2, T1)
    X1, X2 = reconstruct(x1, x2, y1, y2, R2, T1)
    if X1 is not None:
        num_sol += 1
        R = R2
        T = T1
        X1_res = X1
        X2_res = X2

    # check (R2, T2)
    X1, X2 = reconstruct(x1, x2, y1, y2, R2, T2)
    if X1 is not None:
        num_sol += 1
        R = R2
        T = T2
        X1_res = X1
        X2_res = X2

    if num_sol == 0:
        print("No valid solution found")
        return None, None, None, None
    elif num_sol == 1:
        print("Unique solution found")
        return R, T, X1_res, X2_res
    else:
        print("Multiple solutions found")
        return R, T, X1_res, X2_res
