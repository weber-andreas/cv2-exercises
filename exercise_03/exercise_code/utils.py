from scipy.signal import convolve2d as conv2
import numpy as np
import cv2


def getGaussiankernel(sigma):
    # compute the 2D Gaussian kernel
    # Input:
    # sigma: float (standard deviation of Gaussian)
    # Output:
    # G: numpy.ndarray (Gaussian kernel) of shape (4*sigma+1, 4*sigma+1)

    k = np.ceil(4 * sigma + 1)
    x = np.arange(-2 * sigma, 2 * sigma + 1)

    ########################################################################
    # TODO:                                                                #
    # Compute the gaussian kernel                                          #
    #                                                                      #
    # Hints:                                                               #
    # - The Gaussian kernel is separable.                                  #
    ########################################################################

    gauss = (
        1
        / np.sqrt(2 * np.pi * sigma)
        * np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2)))
    )

    G_unnormalized = np.outer(gauss, gauss)
    G = G_unnormalized / np.sum(G_unnormalized)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return G


def getGradients(I, sigma=2):
    # compute spatial gradient
    # Input:
    # I: numpy.ndarray (image) of shape (H, W)
    # sigma: float (standard deviation of Gaussian)
    # Output:
    # Ix: numpy.ndarray (image gradient) of shape (H, W)
    # Iy: numpy.ndarray (image gradient) of shape (H, W)

    I_ = I.copy()
    I_ = I_ / 255.0
    if sigma > 0:
        k = int(np.ceil(4 * sigma + 1))
        I_ = cv2.GaussianBlur(
            I_,
            (k, k),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_DEFAULT,
        )

    ########################################################################
    # TODO:                                                                #
    # Compute the spatial gradient using central differences.              #
    ########################################################################
    Ix = np.zeros_like(I_)
    Iy = np.zeros_like(I_)

    Ix[:, 1:-1] = (I_[:, 2:] - I_[:, :-2]) / 2
    Iy[1:-1, :] = (I_[2:, :] - I_[:-2, :]) / 2

    # Forward/backward differences at borders
    Ix[:, 0] = (I_[:, 1] - I_[:, 0]) / 2
    Ix[:, -1] = (I_[:, -1] - I_[:, -2]) / 2
    Iy[0, :] = (I_[1, :] - I_[0, :]) / 2
    Iy[-1, :] = (I_[-1, :] - I_[-2, :]) / 2

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return Ix, Iy


def getTemporalPartialDerivative(I1, I2, sigma=2):
    # compute temporal gradient
    # Input:
    # I1: numpy.ndarray (image) of shape (H, W)
    # I2: numpy.ndarray (image) of shape (H, W)
    # sigma: float (standard deviation of Gaussian)
    # Output:
    # It: numpy.ndarray (temporal gradient) of shape (H, W)

    I1_ = I1.copy()
    I2_ = I2.copy()
    I1_ = I1_ / 255.0
    I2_ = I2_ / 255.0
    if sigma > 0:
        k = int(np.ceil(4 * sigma + 1))
        # blur images with Gaussian kernel
        I1_ = cv2.GaussianBlur(I1_, (k, k), 0, 0, cv2.BORDER_DEFAULT)
        I2_ = cv2.GaussianBlur(I2_, (k, k), 0, 0, cv2.BORDER_DEFAULT)

    ########################################################################
    # TODO:                                                                #
    # Compute the temporal gradient with forward differences               #
    ########################################################################
    It = np.zeros_like(I1_)
    It[:, :-1] = (I2_[:, 1:] - I1_[:, :-1]) / 2
    It[:, -1] = (I2_[:, -1] - I1_[:, -1]) / 2  # backward difference at the last column

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return It


def getM(Ix, Iy, sigma=7):
    # compute structure tensor
    # Input:
    # Ix: numpy.ndarray (image gradient) of shape (H, W)
    # Iy: numpy.ndarray (image gradient) of shape (H, W)
    # sigma: float (standard deviation of Gaussian)
    # Output:
    # M: numpy.ndarray (structure tensor) of shape (H, W, 2, 2)

    ########################################################################
    # TODO:                                                                #
    # Compute the structure tensor M for each pixel.                       #
    #                                                                      #
    # Hints:                                                               #
    # - Use the Gaussian kernel (from  getGaussiankernel)                  #
    #   to compute the structure tensor.                                   #
    # - Use the conv2 function to compute the convolution.                 #
    # - The structure tensor is a 2x2 matrix for each pixel.               #
    # compute the elements of the structure tensor M for each pixel.       #
    # M11: numpy.ndarray (element of structure tensor)                     #
    # M12: numpy.ndarray (element of structure tensor)                     #
    # M22: numpy.ndarray (element of structure tensor)                     #
    ########################################################################

    # Grad I * Grad I.T
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    G = getGaussiankernel(sigma)

    M11 = conv2(Ix2, G, "same", boundary="symm")
    M12 = conv2(Ixy, G, "same", boundary="symm")
    M22 = conv2(Iy2, G, "same", boundary="symm")

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    # note: we do not need to compute M21 separately, since M is symmetrical and therefore M21 == M12
    M = np.stack((M11, M12, M12, M22), axis=-1)
    # create a 2x2 matrix for each pixel
    M = M.reshape(M.shape[0], M.shape[1], 2, 2)
    assert np.allclose(M[:, :, 0, 1], M12)
    assert np.allclose(M[:, :, 1, 0], M12)
    assert np.allclose(M[:, :, 0, 0], M11)
    assert np.allclose(M[:, :, 1, 1], M22)

    return M


def getq(It, Ix, Iy, sigma=7):
    # compute q vector
    # Input:
    # It: numpy.ndarray (temporal gradient) of shape (H, W)
    # Ix: numpy.ndarray (image gradient) of shape (H, W)
    # Iy: numpy.ndarray (image gradient) of shape (H, W)
    # sigma: float (standard deviation of Gaussian)
    # Output:
    # q: numpy.ndarray (q vector) of shape (H, W, 2)

    ########################################################################
    # TODO:                                                                #
    # Compute the tensor q for each pixel.                                 #
    #                                                                      #
    # Hints:                                                               #
    # - Use the Gaussian kernel (from  getGaussiankernel)                  #
    #   to compute the structure tensor.                                   #
    # - Use the conv2 function to compute the convolution.                 #
    # - The structure tensor is a 2-dimensional vector for each pixel.     #
    # compute the elements of the tensor q for each pixel.                 #
    # q1: numpy.ndarray (element of tensor q)                              #
    # q2: numpy.ndarray (element of tensor q)                              #
    ########################################################################

    Itx = It * Ix
    Ity = It * Iy

    G = getGaussiankernel(sigma)

    q1 = conv2(Itx, G, "same", boundary="symm")
    q2 = conv2(Ity, G, "same", boundary="symm")

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    # create a vector for each pixel
    q = np.stack((q1, q2), axis=-1)
    assert np.allclose(q[:, :, 0], q1)
    assert np.allclose(q[:, :, 1], q2)

    return q
