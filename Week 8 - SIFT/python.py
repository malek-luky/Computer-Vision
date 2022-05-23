import math
from scipy.spatial.transform import Rotation
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import sys

sys.path.append('..') #go back 1 folder
from library import *

#################################

SHOW_PLOT = 0
sigma = 1
n = 6
threshold = 25

# Load image
img = cv2.imread('sunflowers.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
if SHOW_PLOT: plt.imshow(gray, cmap="gray")

def plot_images(pic):
    fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(26, 6))
    ax[0].imshow(pic[0])
    ax[1].imshow(pic[1])
    ax[2].imshow(pic[2])
    ax[3].imshow(pic[3])
    ax[4].imshow(pic[4])
    ax[5].imshow(pic[5])
    ax[6].imshow(pic[6])

def scaleSpaced(im, sigma, n):
    blurred_imgs = []
    for i in range(n):
        blur, _, _ = gaussianSmoothing(im, sigma*2**i)
        blurred_imgs.append(blur)
    return np.array(blurred_imgs)

blurred_imgs = scaleSpaced(gray, 1, 8)
if SHOW_PLOT:
    plot_images(blurred_imgs)

#################################

def differenceOfGaussian(im, sigma, n):
    blurred_imgs = scaleSpaced(im, sigma, n)
    DoG = [np.int32(blurred_imgs[i + 1]) - np.int32(blurred_imgs[i])
           for i in range(n - 1)]
    return np.array(DoG)

DoG = differenceOfGaussian(gray,2, 8)
print("Max valued for DoG are decresing: ",np.max(DoG[0]), np.max(DoG[1]), np.max(
    DoG[2]), np.max(DoG[3]), np.max(DoG[4]), np.max(DoG[5]),np.max(DoG[6]))
if SHOW_PLOT:
    plot_images(DoG)

#################################

def max_kernel(matrix,thresh):
    # Return the x,y pixel value if its max in 3x3 neigborhood, otherwise 0
    idx = np.argmax(matrix)
    return idx // 9, idx%9//3, idx%3

def detectBlobs(im, sigma, n,threshold):
    # Return list of blobs in format [(x,y,n),...]
    ret = []
    kernel = 4 #6x6x6 must be lower than n-1
    DoG = differenceOfGaussian(im,sigma, n)
    for z in range(len(DoG)//kernel):
        for y in range(len(im)//kernel): #-kernel so we dont detect the boarder while the image is rotated
            for x in range(len(im[0])//kernel):
                matrix = np.array(DoG[(kernel*z):(kernel*z+kernel), (kernel*y):(kernel*y+kernel), (kernel*x):(kernel*x+kernel)])
                if np.max(matrix)>threshold:
                    x_ret, y_ret,z_ret = max_kernel(matrix, threshold)
                    ret.append((kernel*x+x_ret, kernel*y+y_ret, kernel*z+z_ret))
    print("Detected blobs:",len(ret))
    print("Maximum possible blobs: ",len(DoG)*len(im)*len(im[0])/(kernel*kernel*kernel))
    return np.array(ret)

blobs = detectBlobs(gray,sigma,n,threshold)
im_copy = img.copy()
b, g, r = cv2.split(im_copy)
im_copy = cv2.merge([r, g, b])
for blob in blobs:
    cv2.circle(img=im_copy, center=tuple((blob[0], blob[1])), radius=blob[2]*4, color=(255, 0, 0), thickness=2)
fig, ax = plt.subplots(1, figsize=(10, 10))
plt.imshow(im_copy)
plt.show()

######################


def transformIm(im, theta, s):
    # Setup
    w, h = im.shape
    y, x = np.meshgrid(
        np.linspace(0, w, w, dtype=int), np.linspace(0, h, h, dtype=int))
    x = x.flatten() * s
    y = y.flatten() * s
    locs = np.vstack((x, y))

    # Rotate
    theta = theta*np.pi/180
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    # Rotate
    rotated = R@locs
    # Scale
    rotated[0, :] = rotated[0, :] - rotated[0, :].min()
    rotated[1, :] = rotated[1, :] - rotated[1, :].min()

    # Create canvas
    x_axis = math.ceil(rotated[0, :].max()-rotated[0, :].min())
    y_axis = math.ceil(rotated[1, :].max()-rotated[1, :].min())
    canvas = np.zeros(shape=(y_axis, x_axis))*im.min()
    canvas[::2] = im.max()  # To break up the flat surface
    # Draw onto canvas
    for i in range(len(im[0])):
        for j in range(len(im[:, i])):
            iteration = i * len(im[:, i]) + j
            xpos = np.int32(rotated[0, iteration])
            ypos = np.int32(rotated[1, iteration])
            color = im[j, i]
            canvas[ypos, xpos] = color

    return np.uint8(canvas)

###################################


# ROTATION NUMERO UNO
rot_im_1 = transformIm(gray, theta=15, s=0.75)
blobs = detectBlobs(rot_im_1, sigma, n, threshold)
for blob in blobs:
    cv2.circle(img=rot_im_1, center=tuple(
        (blob[0], blob[1])), radius=blob[2]*4, color=(255, 0, 0), thickness=2)
fig, ax = plt.subplots(1, figsize=(10, 10))
plt.imshow(rot_im_1, cmap="gray")
plt.show()

# ROTATION NUMERO DUE
rot_im_2 = transformIm(gray, theta=-15, s=0.75)
blobs = detectBlobs(rot_im_2, sigma, n, threshold)
for blob in blobs:
    cv2.circle(img=rot_im_2, center=tuple(
        (blob[0], blob[1])), radius=blob[2]*4, color=(255, 0, 0), thickness=2)
fig, ax = plt.subplots(1, figsize=(10, 10))
plt.imshow(rot_im_2, cmap="gray")
plt.show()
