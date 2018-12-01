import numpy as np
import math

def mse(A, B):
    """
    calculate Mean Square Error between 2 matrices
    :param A, B. Input matrices
    """
    return (np.square(A - B)).mean(axis=None)

def noise_attack(image, k):
    """
    source: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    add noise to image
    :param image. input image
    :param k. noise strength
    """
    mean = 0
    if(len(image.shape) == 3):
        row,col,ch= image.shape
        gauss = np.random.normal(mean,1,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
    elif (len(image.shape) == 2):
        row, col = image.shape
        gauss = np.random.normal(mean, 1, (row, col))
        #var = 0.1
       #sigma = var**0.5
        gauss = gauss.reshape(row, col)

    noisy = image + k*gauss
    return noisy

def psnr(A, B):
    """
    compare peak signal-to-noise ratio of 2 images
    :param A, B. input images
    source: https://tutorials.techonical.com/how-to-calculate-psnr-value-of-two-images-using-python/
    """
    meanSE = mse(A,B)
    if meanSE == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(meanSE))

if __name__ == '__main__':
    A = np.random.rand(4,4)
    B = np.random.rand(4,4)
    result = mse(A,B)
    print("Result: ", result)
    noise_attack(A, 1)
    print("psnr: ", psnr(A,B))