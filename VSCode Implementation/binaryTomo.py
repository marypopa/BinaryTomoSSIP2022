import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
import cv2
import random
import math
import sys
import time

#region Utility functions
def fillRandomly(size):
    randomImage = np.zeros((size, size))
    dotNo = np.random.randint(0, size*size)

    for i in range (dotNo):
         xCoord, yCoord = np.random.randint(0, size, 2)
         randomImage[xCoord, yCoord] = 1

    return randomImage

def uniformRegularization():
    return
#endregion

#region SA with an exponential cooling algorithm
def schedule(t, a=50, b=0.005, limit=0.000001):
    T = math.exp(-b*t)*a
    if T < limit:
        return 0
    else:
        return T


def simulated_annealing(sinogram, angles):
    # image to be adjusted
    testImage = np.zeros((sinogram.shape[0], sinogram.shape[0]))
    toIterate = np.arange(0, testImage.shape[0], 1)
    radonImage = radon(testImage, theta=angles)
    c_old = np.linalg.norm(radonImage - sinogram)
    c_start = c_old
    r_objective = 0.00001

    for t in range(1, sys.maxsize):
        T = schedule(t)

        if T == 0:
            return testImage

        for i in np.nditer(toIterate):

            xCoord, yCoord = np.random.randint(0, testImage.shape[0], 2)
            testImage[xCoord, yCoord] = 1 - testImage[xCoord, yCoord]

            radonImage = radon(testImage, theta=angles)
            testSinogram = radonImage - sinogram
            c_new = np.linalg.norm(testSinogram)

            delta_c = c_new - c_old

            if delta_c < 0 or np.exp(-delta_c/T) > random.uniform(0,1):
                c_old = c_new
            else:
                testImage[xCoord, yCoord] = 1 - testImage[xCoord, yCoord]
        

        print(T)
        if c_old/c_start < r_objective:
            return testImage
#endregion

#region SA with other cooling algorithms
def linear_coolingSA(sinogram, angles):
    # image to be adjusted
    #testImage = np.zeros((sinogram.shape[0], sinogram.shape[0]))
    testImage = fillRandomly(sinogram.shape[0])
    toIterate = np.arange(0, testImage.shape[0], 1)
    radonImage = radon(testImage, theta=angles)
    c_old = np.linalg.norm(radonImage - sinogram)
    c_start = c_old
    print(c_old)
    r_objective = 0.00001
    T = 8
    T_factor = 0.98
    T_min = 10**(-14)

    while T > T_min:

        for i in np.nditer(toIterate):

            xCoord, yCoord = np.random.randint(0, testImage.shape[0], 2)
            testImage[xCoord, yCoord] = 1 - testImage[xCoord, yCoord]

            radonImage = radon(testImage, theta=angles)
            testSinogram = radonImage - sinogram
            c_new = np.linalg.norm(testSinogram)

            delta_c = c_new - c_old

            if delta_c < 0 or np.exp(-delta_c/T) > random.uniform(0,1):
                c_old = c_new
            else:
                testImage[xCoord, yCoord] = 1 - testImage[xCoord, yCoord]
        
        T = T * T_factor

        if c_old/c_start < r_objective:
            print('finished early')
            return testImage

    return testImage
#endregion

start_time = time.time()
fig = plt.figure(figsize=(10, 7))

# can change the image to image array
image = cv2.imread("barn-owl.png", cv2.IMREAD_GRAYSCALE)
image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

image = cv2.resize(image, (32, 32))
image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

# number of projections
projectionNo = 20

# projection angles
angles = np.linspace(0, 180, projectionNo)

# core sinogram
sinogram = radon(image, theta=angles)

testImage = simulated_annealing(sinogram, angles)
#testImage = linear_coolingSA(sinogram, angles)
end_time = time.time()

fig.add_subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')


fig.add_subplot(1, 2, 2)
plt.imshow(testImage, cmap='gray')
plt.title('Reconstructed')

plt.show()
print(end_time - start_time)

