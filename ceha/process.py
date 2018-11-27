"""
This module contains the entire process of compressed sensing as proposed
by the paper
"""

import cv2
import sys
import logging
import compress.py as cp
import random_exchange.py as re
import numpy as np
from pyCSalgos import SmoothedL0 as sl

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

def read_into_blocks(img):
    img_height, img_width = img.shape[:2]
    #divide the array into 4 blocks, block 1 is at top left corner, going counter-clockwise
    row_idx = img_height//2
    col_idx = img_width//2
    b1 = img[:row_idx, :col_idx].copy()
    b2 = img[row_idx:, :col_idx].copy()
    b3 = img[row_idx:, col_idx:].copy()
    b4 = img[:row_idx, col_idx:].copy()
    return (b1,b2,b3,b4)

def compress_encrypt():
    """
    compress and encrypt input image
    """

    #read image from command line
    img = cv2.imread(sys.argv[1])
    img_width, img_height = img.shape[:2]

    #Step 1: read and divide into blocks
    blocks = read_into_blocks(img)

    #Step 2: measure blocks with phi_matrix
    lamb = 2
    log_map1 = cp.logistic_map(0.11, img_width)
    log_map2 = cp.logistic_map(0.23, img_width)
    phi1 = cp.phi_matrix(log_map1, lamb)
    phi2 = cp.phi_matrix(log_map2, lamb)
    c1 = phi1.dot(blocks[0])
    c3 = phi1.dot(blocks[2])
    c2 = phi2.dot(blocks[1])
    c4 =  phi2.dot(blocks[3])

    #Step 3: pixel exchange
    r1 = R_matrix(phi1, lamb)
    r2 = R_matrix(phi2, lamb)
    #e1 is block1 after exchange
    e1, e2 = re.rand_pixel_exchange(c1, c2, r1)
    e2, e3 = re.rand_pixel_exchange(c2, c3, r2)
    e3, e4 = re.rand_pixel_exchange(c3, c4, r1)
    e4, e1 = re.rand_pixel_exchange(c4, c1, r2)

    #construct encrypted image
    return np.bmat([[e1, e2], [e3, e4]])

def decompress_decrypt():
    #Step 1: read into blocks 
    img = cv2.imread(sys.argv[1])
    c1, c2, c3, c4 = read_into_blocks(img)

    #Step 2: inverse pixel exchange
    d4, d1 = re.rand_pixel_exchange(c4, c1, r2)
    d3, d4 = re.rand_pixel_exchange(c3, c4, r1)
    d2, d3 = re.rand_pixel_exchange(c2, c3, r2)
    d1, d2 = re.rand_pixel_exchange(c1, c2, r1)

    #Step 3: run SL0 algorithm with phi matrix


    #Step 4: constrct the decripted image

if __name__ == '__main__':
    result = read_into_blocks()
    print("first block size: ", result[0].shape[:2][0], " ", result[0].shape[:2][1])
    print("second block size: ", result[1].shape[:2][0], " ", result[1].shape[:2][1])
    print("third block size: ", result[2].shape[:2][0], " ", result[2].shape[:2][1])
    print("fourth block size: ", result[3].shape[:2][0], " ", result[3].shape[:2][1])