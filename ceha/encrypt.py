"""Module controlling image encryption and decryption as well as associated
algorithms.

No need to directly call this, can be instead utilized using user friendly
process module.

"""

import numpy as np
import math

##Generate a random matrix of given sizes,
##pixel values are uniformly distributed in range [0,1)
def R_matrix(phi_matrix, lamb):
	return phi_matrix/lamb
	
def rand_pixel_exchange(block1, block2, R, mode):
    """Random pixel exchange between 2 blocks.

    :param block1: First block of pixels.
    :param block2: Second block of pixels.
    :param R: Random matrix.

    """

    R_height, R_width = R.shape[:2]

    b1 = block1.copy()
    b2 = block2.copy()

    R_bar = R.mean()

    if mode == 'encrypt':
        range1 = range(R_height)
        range2 = range(R_width)
    elif mode == 'decrypt':
        range1 = list(reversed(range(R_height)))
        range2 = list(reversed(range(R_width)))
    else:
        raise Exception('Unkown mode type {}'.format(mode))

    for m in range1:
        for n in range2:
            new_m = int(round((R_height - 1) * math.sin(math.pi * R[m,n])))
            new_n = int(round((R_width - 1) * R[m,n]))
            if R[m,n] > R_bar:
                b1[new_m,new_n], b2[m,n] = b2[m,n], b1[new_m,new_n]
                b1[m,n], b2[new_m,new_n] = b2[new_m,new_n], b1[m,n]
            else:
                b1[new_m,new_n], b1[m,n] = b1[m,n], b1[new_m,new_n]
                b2[new_m,new_n], b2[m,n] = b2[m,n], b2[new_m,new_n]
    return (b1, b2)


if __name__ == '__main__':
    phi_matrix = np.random.rand(4,4)
    R = R_matrix(phi_matrix, 2)
    print("Random Matrix R")
    print(R)
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    print("Matrix A: ")
    print(A)
    print("Matrix B: ")
    print(B)
    r = rand_pixel_exchange(A, B, R)
    print("Result matrix A: ")
    print( r[0])
    print("Result matrix B: " )
    print(r[1])
