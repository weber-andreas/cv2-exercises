import numpy as np
import math
from abc import ABC, abstractmethod


def compute_relative_pose(pose_1: np.ndarray, pose_2: np.ndarray) -> np.ndarray:
    """
    Inputs:
    - pose_i transform from cam_i to world coordinates, matrix of shape (3,4)
    Outputs:
    - pose transform from cam_1 to cam_2 coordinates, matrix of shape (3,4)
    """

    ########################################################################
    # TODO:                                                                #
    # Compute the relative pose, which transform from cam_1 to cam_2       #
    # coordinates.                                                         #
    ########################################################################

    # Xw = R1 @ X1 + t1 = R2 @ X2 + t2
    # X2 = R2.T @ R1 @ X + R2.T @ (t1 - t2)
    # X2 = R12 @ X1 + R2.T @ (t2 - t1)

    R1 = pose_1[:, :3]
    t1 = pose_1[:, 3]
    R2 = pose_2[:, :3]
    t2 = pose_2[:, 3]

    R12 = R2.T @ R1
    t12 = R2.T @ (t1 - t2)

    pose = np.hstack((R12, t12.reshape(3, 1)))
    assert pose.shape == (3, 4), "The pose should be a 3x4 matrix."
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return pose


class Camera(ABC):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    @abstractmethod
    def project(self, pt):
        """Project the point pt onto a pixel on the screen"""

    @abstractmethod
    def unproject(self, pix, d):
        """Unproject the pixel pix into the 3D camera space for the given distance d"""


class Pinhole(Camera):

    def __init__(self, w, h, fx, fy, cx, cy):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def project(self, pt):
        """
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the pinhole model, vector of size 2
        """
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the pinhole model.                 #
        ########################################################################

        # lambda * x = K @ X
        # x = K @ Xw / lambda, labda = x_z

        pix = self.K @ pt
        pix = pix[:2] / pix[2]  # Normalize by the third coordinate: lambda = Z

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        """
        Inputs:
        - pix, vector of size 2
        - d, scalar
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the pinhole model, vector of size 3
        """
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the pinhole#
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################

        # lambda * x = K @ X
        # X = lambda * (K^-1 @ x)
        pix_homogeneous = np.array([pix[0], pix[1], 1])
        K_inv = np.linalg.inv(self.K)
        ray = K_inv @ pix_homogeneous
        ray /= np.linalg.norm(ray)  # Normalize the direction vector
        final_pt = ray * d

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return final_pt


class Fov(Camera):

    def __init__(self, w, h, fx, fy, cx, cy, W):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.W = W

    def project(self, pt):
        """
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the Fov model, vector of size 2
        """
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the Fov model.                     #
        ########################################################################

        x, y, z = pt
        assert z > 0, "The depth (z-coordinate) must be positive for projection."

        pt_proj = np.array([x / z, y / z])

        def g_atan(r):
            return 1 / (self.W * r) * math.atan(2 * r * math.tan(self.W / 2))

        # Function to compute the angle in radians
        pi_d = g_atan(np.linalg.norm(pt_proj)) * pt_proj
        x = self.K @ np.array([pi_d[0], pi_d[1], 1])
        pix = x[:2] / x[2]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        """
        Inputs:
        - pix, vector of size 2
        - d, scalar
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the Fov model, vector of size 3
        """
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the FOV    #
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################

        u, v = pix
        pix_homogeneous = np.array([u, v, 1])

        def f_atan(r):
            return math.tan(r * self.W) / (2 * r * math.tan(self.W / 2))

        K_inv = np.linalg.inv(self.K)
        direction = K_inv @ pix_homogeneous
        pi_d = direction[0:2] / direction[2]  # Normalize by the third coordinate
        pi = f_atan(np.linalg.norm(pi_d)) * pi_d

        ray = np.array([pi[0], pi[1], 1])
        ray /= np.linalg.norm(ray)
        final_pt = ray * d

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return final_pt
