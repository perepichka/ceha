'''

WRITER: SHAKIL AHMED
Input: Logistic map of size N.
Output: an 2N*N circular matrix.

'''

import numpy as np


def phi_matrix(log_map, lamb):
	n = len(log_map)
	pm = np.zeros((2*n,n))

	m = 2 * n
	pm[0,:] = log_map
	for i in range(1,m):
		pm[i,0] = pm[i-1,n-1] * lamb
		pm[i,1:n] = pm[i-1,0:n-1]
	return pm


if __name__ == "__main__":
    
	a = np.array([1,2,3,4])

	pm = phi_matrix(a,.2)
	print(pm)





