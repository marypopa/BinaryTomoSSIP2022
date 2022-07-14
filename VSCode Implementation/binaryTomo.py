import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
import cv2
import random
import math
import sys
import time

#region Utility functions
def binarizeWithSize(imageName, size):
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

    image = cv2.resize(image, (size, size))
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

    return image


def fillRandomly(size):
    randomImage = np.zeros((size, size))
    # dotNo = np.random.randint(0, size*size)

    # for i in range (dotNo):
    #      xCoord, yCoord = np.random.randint(0, size, 2)
    #      randomImage[xCoord, yCoord] = 1

    return randomImage


def uniformRegularization(image):
    totalSum = 0
    size = image.shape[0]

    for i in range(size*size):
        xCoord = i // size
        yCoord = i % size
        pixelSum = 0

        # if xCoord - 1 >= 0 and yCoord - 1 >= 0:
        #     pixelSum += abs(image[xCoord, yCoord] - image[xCoord - 1, yCoord - 1]) 
        
        # if xCoord - 1 >= 0:
        #     pixelSum += abs(image[xCoord, yCoord] - image[xCoord - 1, yCoord]) 

        # if xCoord - 1 >= 0 and yCoord + 1 < size:
        #     pixelSum += abs(image[xCoord, yCoord] - image[xCoord - 1, yCoord + 1])

        # if yCoord - 1 >= 0:
        #     pixelSum += abs(image[xCoord, yCoord] - image[xCoord, yCoord - 1])

        if yCoord + 1 < size:
            pixelSum += (image[xCoord, yCoord] - image[xCoord, yCoord + 1])**2

        # if xCoord + 1 < size and yCoord - 1 >= 0:
        #     pixelSum += abs(image[xCoord, yCoord] - image[xCoord + 1, yCoord - 1])

        if xCoord + 1 < size:
            pixelSum += (image[xCoord, yCoord] - image[xCoord + 1, yCoord])**2
        
        # if xCoord + 1 < size and yCoord + 1 < size:
        #     pixelSum += abs(image[xCoord, yCoord] - image[xCoord + 1, yCoord + 1])

        #pixelSum = pixelSum / (9*255)

        totalSum += pixelSum / 255
    return totalSum


def comparisonReport(original, reconstruction):
    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives
    tn = 0  # true negatives
    whiteNo = 0

    for i in range(original.shape[0]):
        for j in range(original.shape[0]):
            trueOriginal = original[i, j]/255

            if trueOriginal == 1:
              whiteNo += 1

            if(trueOriginal - reconstruction[i, j]) == 0:
                if trueOriginal == 0: 
                    tp += 1
                else:
                    tn += 1
            else:
                if trueOriginal == 1:
                    fn += 1 
                else:
                    fp += 1

    recall = tp/(tp + fn)
    misguessed = fn + fp

    rme = misguessed/whiteNo

    print("Recall: ", recall)
    print("Error: ", rme*100 , " %")


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
    regularizationParam = 14
    c_old = np.linalg.norm(radonImage - sinogram) + regularizationParam * uniformRegularization(testImage)
    c_start = c_old
    r_objective = 0.00001
    T = 4
    T_factor = 0.97
    T_min = 10**(-14)

    while T > T_min:

        for i in np.nditer(toIterate):

            xCoord, yCoord = np.random.randint(0, testImage.shape[0], 2)
            testImage[xCoord, yCoord] = 1 - testImage[xCoord, yCoord]

            radonImage = radon(testImage, theta=angles)
            testSinogram = radonImage - sinogram
            c_new = np.linalg.norm(testSinogram) + regularizationParam * uniformRegularization(testImage)

            delta_c = c_new - c_old

            if delta_c < 0 or np.exp(-delta_c/T) > random.uniform(0,1):
                c_old = c_new
            else:
                testImage[xCoord, yCoord] = 1 - testImage[xCoord, yCoord]
        
        T = T * T_factor

        if c_old/c_start < r_objective:
            return testImage

    return testImage
#endregion


start_time = time.time()
fig = plt.figure(figsize=(10, 7))

image = binarizeWithSize("barn-owl.png", 64)

np.random.seed(107)
# number of projections
projectionNo = 20

# projection angles
angles = np.linspace(0, 180, projectionNo)

# core sinogram
sinogram = radon(image, theta=angles)

# noise
noise = np.random.normal(0, 1, (sinogram.shape[0], sinogram.shape[1]))
sinogram = sinogram + noise

reconstruction = linear_coolingSA(sinogram, angles)
end_time = time.time()

fig.add_subplot(1, 2, 1)
# plt.imshow(sinogram, cmap='gray')
plt.imshow(image, cmap='gray')
plt.title('Original')

fig.add_subplot(1, 2, 2)
plt.imshow(reconstruction, cmap='gray')
# plt.imshow(sinogramWNoise, cmap='gray')

plt.title('Reconstructed')
work_time = end_time - start_time

comparisonReport(image, reconstruction)
plt.show()

print(work_time//60, ' minutes and ', work_time%60, ' seconds' )
