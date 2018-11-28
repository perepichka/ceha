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
	
"""
random pixel exchange between 2 blocks
b1, b2 are two blocks of matrix
R is a random matrix
Output: 2 blocks of matrix after pixel exchange
"""
def rand_pixel_exchange(b1, b2, R):
    R_width, R_height = R.shape[:2]

    #calculate sum of all values in R
    sum = 0
    for m in range(0, R_height - 1):
    for n in range(0, R_width - 1):
    sum = sum + R[m][n]

    R_bar = 1/(R_width*R_height)*sum

    for m in range(0, len(b1) - 1):
        for n in range(0, len(b1[0]) - 1):
            #since the range of m and n in the paper starts from 1,
            #the following formula is a bit different from the one in the paper
            new_m = 1 + int(round((R_height - 2) * math.sin(math.pi * R[m][n])))
            new_n = 1 + int(round((R_width - 2) * R[m][n]))
            if R[m][n] > R_bar:
                temp = b1[new_m][new_n]
                b1[new_m][new_n] = b2[m][n]
                b2[m][n] = temp
                temp = b2[new_m][new_n]
                b2[new_m][new_n] = b1[m][n]
                b1[m][n] = temp
            else:
                temp = b1[new_m][new_n]
                b1[new_m][new_n] = b1[m][n]
                b1[m][n] = temp
                temp = b2[new_m][new_n]
                b2[new_m][new_n] = b2[m][n]
                b2[m][n] = temp
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
