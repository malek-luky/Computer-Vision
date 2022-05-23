import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import math
import sys
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize
import time
from PIL import Image
import random
from copy import deepcopy

##################################################
# WEEK 1
##################################################


def box3d(n):
    x = []
    y = []
    z = []

    for i, val in enumerate([-0.5, -0.5, 0, 0.5, 0.5], 1):
        x = np.concatenate((x, val*np.ones(n)))
        y = np.concatenate((y, np.linspace(-0.5, 0.5, n)))
        z = np.concatenate((z, (2*(i % 2)-1)*val*np.ones(n)))
    x2 = np.concatenate((x, y))
    x2 = np.concatenate((x2, x))
    y2 = np.concatenate((y, z))
    y2 = np.concatenate((y2, z))
    z2 = np.concatenate((z, x))
    z2 = np.concatenate((z2, y))
    return np.vstack((x2, y2, z2))


def to_inhomogeneous_3D(Q):
    return Q[0:3, :]/Q[3, :]


def to_inhomogeneous(q):
    return q[0:2, :]/q[2, :]


def to_homogeneous(q):
    return np.vstack((q, np.ones(len(q[0]))))


def project_points(K, R, t, Q):
    # p_h = K*[R t]*Q
    return K@np.hstack((R, t))@Q

##################################################
# WEEK 2
##################################################


def eucl_dist(p):
    return np.sqrt(np.square(p[0, :]) + np.square(p[1, :]))


def delta_r(p, dist):
    p = eucl_dist(p)
    return dist[0]*p**2 + dist[1]*p**4 + dist[2]*p**8


def radial_dist(p, dist):
    return p*(1+delta_r(p, dist))


def project_points_dist(K, R, t, Q, dist):
    RtQ_homo = np.hstack((R, t))@Q
    RtQ_inhomo = RtQ_homo[0:2, :]/RtQ_homo[2, :]
    rad_dist = radial_dist(RtQ_inhomo, dist)
    rad_dist_homo = np.vstack((rad_dist, np.ones(len(rad_dist[0]))))
    return K@rad_dist_homo


def undistortImage(gray, K, dist):
    """
    Undistorts an image by mapping the colors a new empty image.
    TODO: Assumes image is one single channel.
    """
    (height, width) = gray.shape
    ratio = height/width
    # Generate meshgrid of all pixels.
    scale = 1.2
    y, x = np.meshgrid(
        np.linspace(-scale, scale, num=width), np.linspace(-scale*ratio, scale*ratio, num=height))
    x = x.flatten()
    y = y.flatten()
    mapping_matrix = np.vstack((x, y)).astype(float)  # [x, y]

    # Distort the grid
    p = radial_dist(mapping_matrix, dist)

    q = np.vstack((p, np.ones(len(p[0]))))
    # Multiply by the camera matrix
    q = K@q
    # Translate back to cartesian coordinates and return (divide all by s, then remove s)
    target_matrix = q[0:2, :]/q[2, :]

    # Target_matrix hold the distorted positions.
    canvas_matrix = np.zeros(shape=(int(height), int(width)), dtype=int)

    for i in range(height):
        for j in range(width):

            row = int(target_matrix[0][i*width + j])
            col = int(target_matrix[1][i*width + j])

            intensity = int(gray[row][col])
            canvas_matrix[i][j] = intensity

    return canvas_matrix


def homography_point(H, q):
    q_ret = H@q
    p_ret = to_inhomogeneous(q_ret)
    return p_ret  # inhomogeneous


def get_b(q1, q2):
    """
    Parameters
    ----------
    q1, q2: 3 x n numpy arrays
        sets of points
    """
    B = B_i(q1, q2, 0)
    for i in range(1, len(q1[0])):
        B = np.vstack((B, B_i(q1, q2, i)))
    return B


def B_i(q1, q2, i):
    return np.kron(q2[:, i], np.array([[0, -1, q1[1, i]], [1, 0, -q1[0, i]], [-q1[1, i], q1[0, i], 0]]))


def hest(q1, q2, norm=False):
    if norm:
        T2, q2 = normalize2d(q2)
        T1, q1 = normalize2d(q1)
    B = get_b(q1, q2)
    _, _, vh = np.linalg.svd(B)
    H = np.reshape(vh[-1], (3, 3), 'F')
    if norm:
        H = np.linalg.inv(T1) @ H @ T2
    return H

def normalize2d(Q):
    mean = np.mean(Q, axis=1)
    std = np.std(Q, axis=1)
    T = np.array([[1/std[0], 0, -mean[0]/std[0]],
                  [0, 1/std[1], -mean[1]/std[1]],
                 [0, 0, 1]])
    return [T, T@Q]


def random_generator(n):
    q = np.array(
        [[random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)]]).T
    for i in range(0, n):
        q = np.hstack((q, np.array(
            [[random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)]]).T))
    return q


def apply_homography(q1, H):
    q2 = np.vstack((homography_point(H, np.array([q1[:, 0]]).T), 1))
    for i in range(1, len(q1[0])):
        q2 = np.hstack(
            (q2, np.vstack((homography_point(H, np.array([q1[:, i]]).T), 1))))
    return q2


##################################################
# WEEK 3
##################################################

def cross0p(p):
    x, y, z = p.reshape(3)
    ret = np.array([
        [0, - z,  y],
        [z,   0, -x],
        [-y,   x,  0],
    ])
    return ret


def essential_matrix(t, R):
    ret = cross0p(t)@R
    return ret


def fundamental_matrix(K1, K2, E):
    # E = essential matrix
    F = np.linalg.inv(K2.T)@E@np.linalg.inv(K1)
    return F


def epipolar_line(F, q1):
    # returns epipolar line of q1 in camera 2
    return F@q1


def triangulate(q, P):  # Credits to Julliete
    """
    Return the traingulation.

    Parameters
    ----------
    q: 3 x n numpy array
        Homogenous pixel coordinates q1... qn
        One for each camera seeing the point.
        At least two.
    P: list of 3 x 4 numpy arrays
        Projection matrices P1... Pn
        For each pixel coordinate

    Return
    ------
    Q: 3 x 1 numpy array
        Triangulation of the point using the linear SVD algorithm
    """
    _, n = q.shape  # n = no. cameras has seen pixel.

    # Prepare B matrix. Two rows for each camera n.
    B = np.zeros((2 * n, 4))
    for i in range(n):
        B[2 * i: 2 * i + 2] = [
            P[i][2, :] * q[0, i] - P[i][0, :],
            P[i][2, :] * q[1, i] - P[i][1, :],
        ]
    # BQ = 0. Minimize using Svd.
    _, _, vh = np.linalg.svd(B)
    Q = vh[-1, :]  # Q is ev. corresponding to the min. singular point.
    return Q[:3].reshape(3, 1) / Q[3]  # Reshape and scale

##################################################
# WEEK 4
##################################################


def DLT(Q, q, normalize=True):
    """Return the projection matrix P such that q = P @ Q.

    Parameters
    ----------
    Q, q: numpy arrays
        Homogeneous points before and after the projection: q = PQ.
    """
    _, n = Q.shape
    if normalize:
        Tq, q = normalize2d(q)

    B = np.zeros((3 * n, 12))
    for i in range(n):
        B[3 * i: 3 * i + 3] = np.kron(Q[:, i], cross0p(q[:, i]))
    u, s, vh = np.linalg.svd(B)
    P = vh[-1]
    P = P.reshape(4, 3).T

    if normalize:
        P = np.linalg.inv(Tq) @ P  # @ TQ

    return P


def checkerboard_points(n, m):  # Julliete
    """Return the inhomogeneous 3D points of a checkerboard."""
    return np.array([
        [i - (n - 1) / 2 for j in range(m) for i in range(n)],
        [j - (m - 1) / 2 for j in range(m) for i in range(n)],
        np.zeros(n * m),
    ])

def hest_ex4(q_before_H, q_after_H):
    """ Dont delete it, it is used in estimateHomographies"""
    #TODO: Make it general, struggling with the q1 = q1/q1[2] line
    T1, q1 = normalize2d(q_after_H)
    T2, q2 = normalize2d(q_before_H)
    q1 = q1/q1[2]
    q2 = q2/q2[2]
    B = get_b(q1, q2)
    u, s, vh = np.linalg.svd(B)
    H = np.reshape(vh[-1, :], (3, 3), 'F')
    H = np.linalg.inv(T1) @ H @ T2
    return H/H[2, 2]

def estimateHomographies(Q_omega, qs):
    """Return homographies that map from Q_omega to each of the entries in qs.

    Parameters
    ----------
    Q_omega: numpy array
        original un-transformed checkerboard points in 3D.
    qs: list of arrays
        each element in the list containing Q_omega projected to the image
        plane from different views.

    Return
    ------
    list of 3 x 3 arrays
        homographies

    """
    m = [0, 1, 3]
    return [hest_ex4(Q_omega[m, :], qs[i]) for i in range(len(qs))]


def estimate_b(Hs):
    """Return the estimate of the b matrix.

    Parameters
    ----------
    Hs: list of 3 x 3 arrays
        homographies
    """
    n = len(Hs)
    V = np.zeros((2 * n, 6))

    for i in range(n):
        V[2 * i: 2 * i + 2] = np.array([
            get_v(0, 1, Hs[i]),
            get_v(0, 0, Hs[i]) - get_v(1, 1, Hs[i])
        ])
    u, s, vh = np.linalg.svd(V)
    b = vh[-1]
    return b


def get_v(alpha, beta, H):
    return np.array([
        H[0, alpha] * H[0, beta],
        H[0, alpha] * H[1, beta] + H[1, alpha] * H[0, beta],
        H[1, alpha] * H[1, beta],
        H[2, alpha] * H[0, beta] + H[0, alpha] * H[2, beta],
        H[2, alpha] * H[1, beta] + H[1, alpha] * H[2, beta],
        H[2, alpha] * H[2, beta],
    ])


def estimateIntrinsics(Hs):
    """Return the camera matrix given a list of homographies.

    Parameters
    ----------
    Hs: list of 3 x 3 numpy arrays
        homographies

    Return
    ------
    A: 3 x 3 numpy array
        Camera matrix
    """
    b = estimate_b(Hs)
#     B = np.array([
#         [b[0], b[1], b[3]],
#         [b[1], b[2], b[4]],
#         [b[3], b[4], b[5]],
#     ])

    term1 = b[1] * b[3] - b[0] * b[4]
    term2 = b[0] * b[2] - b[1] ** 2

    v0 = term1 / term2
    lambda_ = b[5] - (b[3] ** 2 + v0 * term1) / b[0]
    alpha = np.sqrt(lambda_ / b[0])
    beta = np.sqrt(lambda_ * b[0] / term2)
    gamma = -b[1] * alpha ** 2 * beta / lambda_
    u0 = gamma * v0 / beta - b[3] * alpha ** 2 / lambda_

    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1],
    ])
    return A


def estimateExtrinsics(K, Hs):
    """Return the extrinsic parameters of the camera."""
    Rs = []
    ts = []
    for H in Hs:
        lambda_ = 1 / np.linalg.norm(np.linalg.inv(K) @ H[:, 0])
        r0 = lambda_ * np.linalg.inv(K) @ H[:, 0]
        r1 = lambda_ * np.linalg.inv(K) @ H[:, 1]
        t = lambda_ * np.linalg.inv(K) @ H[:, 2]

        Rs.append(np.vstack((r0, r1, np.cross(r0, r1))).T)
        ts.append(t.reshape(3, 1))  # ts.append(t.T)
    return Rs, ts


def calibrateCamera(qs, Q):
    """Return the intrisic and extrinsic parameters of a camera.

    Based on several view of a set of points.

    Parameters
    ----------
    qs: list of arrays
        each element in the list containing Q projected to the image
        plane from different views.
    Q: numpy array
        original un-transformed checkerboard points in 3D.
    """
    Hs = estimateHomographies(Q_omega=Q, qs=qs)
    K = estimateIntrinsics(Hs)
    Rs, ts = estimateExtrinsics(K, Hs)
    return K, Rs, ts


##################################################
# WEEK 5
##################################################

def triangulate_nonlin(q, P):
    """Triangulate using nonlinear optimisation.

    Parameters
    ----------
    q: 3 x n numpy array
        Pixel coordinates q1... qn
    P: list of 3 x 4 numpy arrays
        Projection matrices P1... Pn

    Return
    ------
    Q: 3 x 1 numpy array
        Triangulation of the point using the linear algorithm
    """
    n = len(P)
    x0 = triangulate(q, P).reshape(3)

    def compute_residuals(Q):
        Q = Q.reshape(3, 1)
        absolute_errors = np.zeros(2 * n)
        for i in range(n):
            qh = P[i] @ np.vstack((Q, 1))
            qr = qh[:2].reshape(2) / qh[2]
            absolute_errors[2 * i: 2 * i + 2] = qr - q[:2, i]
        return absolute_errors
    res = scipy.optimize.least_squares(compute_residuals, x0)
    return res["x"].reshape(3, 1)


def get_rgb(path):
    bgr_img = cv2.imread(path)
    b, g, r = cv2.split(bgr_img)       # get b,g,r
    image = cv2.merge([r, g, b])
    return image


def get_reprojection_images(Q, Is, K, Rs, ts, distortion_coeff=[0, 0, 0]):
    reprojection = []
    qs = []
    for i, im in enumerate(Is):
        im_copy = im.copy()
        # Projection
        q = project_points_dist(K, Rs[i], ts[i], Q, distortion_coeff)
        qs.append(q)
        # Draw all points onto images
        for k in range(len(q[0])):
            cv2.circle(
                img=im_copy,
                center=(int(q[0, k]), int(q[1, k])),
                radius=1,
                color=(255, 0, 0),
                thickness=4
            )
        reprojection.append(im_copy)
    return qs, reprojection


##################################################
# WEEK 6
##################################################

def gaussian1DKernel(sigma, rule=5, eps=0):
    """
    Returns 1D filter kernel g, and its derivative gx.
    """
    if eps:
        filter_size = eps
    else:
        filter_size = np.ceil(sigma*rule)
    x = np.arange(-filter_size, filter_size+1)  # filter
    # Make kernel
    g = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-x**2 / (2*sigma**2))
    # g /= g.sum() # Normalize filter to 1. No need with normalization factor
    g = g.reshape(-1, 1)  # Make it into a col vector
    # Make the derivate of g.
    # NB! Need the normalization term of the gaussian
    gx = -(-x**2)/(sigma**2) * g[:, 0]
    gx = gx.reshape(-1, 1)  # Make it into a col vector
    return g, gx, x


def gaussianSmoothing(im, sigma):
    """
    Returns the gaussian smoothed image I, and the image derivatives Ix and Iy.
    """
    # 1 obtain the kernels for gaussian and for differentiation.
    g, gx, _ = gaussian1DKernel(sigma=sigma)
    # 2 Filter the image in both directions and diff in both directions
    I = cv2.filter2D(cv2.filter2D(im, -1, g), -1,
                     g.T)  # smooth I = g * g.T * I
    # 3 Differentiate - d/dx I = g * gx.T * I
    Ix = cv2.filter2D(cv2.filter2D(im, -1, gx.T), -1, g)
    Iy = cv2.filter2D(cv2.filter2D(im, -1, g.T), -1, gx)
    return I, Ix, Iy


def smoothedHessian(im, sigma, epsilon):
    """
    Calculates smooth/average hessian of image.
    Sigma is the width used to calculate the derivatives, while epsilon is used for the kernel inside Hessian.
    """
    _, Ix, Iy = gaussianSmoothing(im, sigma=sigma)
    g_e, _, _ = gaussian1DKernel(sigma=sigma, eps=epsilon)

    C = np.array([[cv2.filter2D(cv2.filter2D(Ix**2, -1, g_e), -1, g_e.T), cv2.filter2D(cv2.filter2D(Ix*Iy, -1, g_e), -1, g_e.T)],
                  [cv2.filter2D(cv2.filter2D(Ix*Iy, -1, g_e), -1, g_e.T), cv2.filter2D(cv2.filter2D(Iy**2, -1, g_e), -1, g_e.T)]])
    return C


def harrisMeasure(im, sigma, epsilon, k):
    C = smoothedHessian(im, sigma=sigma, epsilon=epsilon)
    a, b, c = C[0, 0, :, :], C[1, 1, :, :], C[0, 1, :, :]
    r = a*b - c**2 - k * (a + b)**2
    return r


def cornerDetector(im, sigma, epsilon, k, tau):
    """
    Returns list of points that are both larger than threshold tau, and local maxima.
    Typically 0.1*rmax < tau < 0.8*rmax.
    """
    r = harrisMeasure(im, sigma=sigma, epsilon=epsilon, k=k)
    under_tau = np.where(r < tau*r.max())
    r[under_tau[0][:], under_tau[1][:]] = 0
    # Non-max supression
    change = True
    while change:
        change = False
        test = np.where(r > 0)
        for k in range(len(test[0])):
            i = test[0][k]
            j = test[1][k]
            try:
                if r[i, j] > r[i+1, j] and r[i, j] >= r[i-1, j] and r[i, j] > r[i, j+1] and r[i, j] >= r[i, j-1] and (r[i+1, j]+r[i-1, j]+r[i, j+1]+r[i, j-1]) > 0:
                    r[i+1, j] = 0
                    r[i-1, j] = 0
                    r[i, j+1] = 0
                    r[i, j-1] = 0
                    change = True
            except IndexError:
                r[i, j] = 0
                change = True

    corners = np.where(r > 0)
    return corners

##################################################
# WEEK 7
##################################################

######################### Exercise 7.1 ############################


def test_points(n_in, n_out):
    # create points with n_in inliers and n_out outliers
    a = (np.random.rand(n_in)-.5)*10  # a=[1 x n_in]
    b = np.vstack((a, a*.5+np.random.randn(n_in)*.25))  # b=[2 x n_in]
    points = np.hstack((b, 2*np.random.randn(2, n_out)))  # points =
    return np.random.permutation(points.T).T


def line_normalizer(a, b, c):
    scale = 1/b
    a *= -scale
    c *= scale
    return (a, c)


def est_line(P, Q):
    # find line from two points, by = ax + c
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a*(P[0]) + b*(P[1])
    a, b = line_normalizer(a, b, c)
    return (a, b)
# def est_line(p1, p2):
#     return np.cross(p1, p2)


def plot_graph(pts, line):
    a = line[0]
    b = line[1]
    plt.plot(pts[0], pts[1], 'ro')
    x = np.linspace(-5., 5.)
    plt.plot(x, a*x+b)
    return plt


def choose_2points(pts):
    list = []
    while len(list) < 2:
        r = np.random.randint(0, len(pts[0])-1)
        if r not in list:
            list.append(r)
    P = (pts[0][int(list[0])], pts[1][int(list[0])])
    Q = (pts[0][int(list[1])], pts[1][int(list[1])])
    return P, Q

######################### Exercise 7.2 ############################


def inlier(R, line, threshold):
    # P = (x,y)
    # y = ax+b
    a = line[0]
    b = line[1]
    y = a*R[0]+b
    dist = abs(R[1]-y)
    # print("Distance from line: ",dist)
    ret = 1 if dist < threshold else 0
    return ret

######################### Exercise 7.3 ############################

def consensus(pts, threshold, line):
    ret = 0
    for i in range(len(pts[0])):
        ret += inlier((pts[0][i], pts[1][i]), line, threshold)
    return ret

######################### Exercise 7.5 ############################

def RANSAC_simple(pts, threshold, iterations):
    best_line = None
    most_inliers = 0
    for i in range(iterations):
        P, Q = choose_2points(pts)
        line = est_line(P, Q)
        total_inliers = consensus(pts, threshold, line)
        if total_inliers > most_inliers:
            most_inliers = total_inliers
            best_line = line
    return most_inliers, best_line

######################### Exercise 7.7 ############################

def pca_line(x):  # assumes x is a (2 x n) array of points
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l@x.mean(1)))
    return l


def remove_outliers(pts, line, threshold):
    pts_inliers_x = []
    pts_inliers_y = []
    for i in range(len(pts[0])):
        if inlier((pts[0][i], pts[1][i]), line, threshold):
            pts_inliers_x.append(pts[0][i])
            pts_inliers_y.append(pts[1][i])
    return np.array([pts_inliers_x, pts_inliers_y])


######################### Exercise 7.8 ############################
def RANSAC(pts, threshold, p):
    best_line = None
    most_inliers = 0
    m = 0
    M = len(pts[0])
    N = 1000  # random value for now
    while (m <= N):
        P, Q = choose_2points(pts)
        line = est_line(P, Q)
        total_inliers = consensus(pts, threshold, line)
        if total_inliers > most_inliers:
            most_inliers = total_inliers
            best_line = deepcopy(line)
            epsilon = 1-most_inliers/M
            N = np.log(1 - p) / np.log(1 - (1 - epsilon**2))
            print("Epsilon:", epsilon, " ==> \tN", N, end="\n")
        m += 1
    print("RANSAC stopped after", m, "Iterations")
    return most_inliers,best_line

######################### WEEK 11 ##################################


def to_inhomogeneous_v2(q):
    """Return inhomogeneous coordinate of point or set of point"""
    q = np.array(q)
    n = q.shape[0]
    q = q[:-1] / q[-1]
    return q.reshape((n-1, -1))


def homogeneous_v2(p):
    """Return homogeneous coordinate of point or set of points."""
    try:
        _, n = p.shape
    except:
        p.reshape(2, 1)
        n = 1
    return np.vstack((p, np.ones(n)))
