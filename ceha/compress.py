"""Module controlling image compression and decompression as well as 
associated algorithms.

No need to directly call this, can be instead utilized using user friendly
process module.

"""

import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

# Values from paper
DEFAULT_MU_MIN = 3.57
DEFAULT_MU_MAX = 4.0
DEFAULT_MU = 3.99

DEFAULT_LAMBDA = 2
DEFAULT_M_RATIO = 3/4


def logistic_map(x_0, n, mu=DEFAULT_MU):
    """Generates logistic map.
    
    :param float x_0: Value of x at zero.
    :param float mu: Value of mu.
    :param float n: Number of iterations.
    
    """

    assert (mu > DEFAULT_MU_MIN and mu < DEFAULT_MU_MAX),\
        'mu is outside of acceptable range, will not yield chaotic log map!'

    m = 2# int(DEFAULT_M_RATIO * n)+1
    x = np.zeros(m*n)

    x[0] = x_0
    for i, prev in enumerate(x):
        if i < m*n-1:
            x[i+1] = mu * prev * (1.0 - prev)
    return x[n:m*n]


def phi_matrix(log_map, lamb=DEFAULT_LAMBDA, m_ratio=None):
    """Generates phi matrix.

    :param str log_map: The log map input.
    :param float lamb: The value for lambda.
    :param int m: The 'm' value.

    """

    n = len(log_map)

    if m_ratio is None:
        m = int(DEFAULT_M_RATIO * n)
    else:
        m = int(m_ratio * n)

    pm = np.zeros((m,n))

    pm[0,:] = log_map
    for i in range(1,m):
        pm[i,0] = pm[i-1,-1] * lamb
        pm[i,1:] = pm[i-1,0:-1]
    return pm


if __name__ == '__main__':

    # Tests logistic map
    #a = logistic_map(0.11, 20)
    a = logistic_map(0.23, 20)

    logging.debug('Logistic map')
    logging.debug(a)
    # Tests phi_matrix
    #a = np.array([1,2,3,4])

    pm = phi_matrix(a,.2)

    logging.debug('Phi Matrix')
    logging.debug(pm)

    # Insert simple test cases for this module here
