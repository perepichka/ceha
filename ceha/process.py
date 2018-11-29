"""
This module contains the entire process of compressed sensing as proposed
by the paper
"""

import cv2
import sys
import logging
import compress as cp
import random_exchange as re
import numpy as np
import SL0 as sl
import os

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
log_map1 = None
log_map2 = None
phi1 = None
phi2 = None
r1 = None
r2 = None
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

    global log_map1, log_map2, phi1, phi2, r1, r2

    #read image from command line
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img_width, img_height = img.shape[:2]
    if len(img.shape) == 3:
        print("image still 3-D")
        #img = img[:,:,0]


    #Step 1: read and divide into blocks
    b1, b2, b3, b4 = read_into_blocks(img)
    print("size of b1: ", b1.shape)

    #Step 2: measure blocks with phi_matrix
    lamb = 2
    log_map1 = cp.logistic_map(0.11, img_width//2)
    log_map2 = cp.logistic_map(0.23, img_width//2)
    phi1 = cp.phi_matrix(log_map1, lamb)
    phi2 = cp.phi_matrix(log_map2, lamb)
    print("size of log_map1: ", log_map1.shape[:2])
    print("size of phi1: ", phi1.shape[:2])
    print("size of b1: ", b1.shape[:2])
    c1 = phi1.dot(b1)
    c3 = phi1.dot(b3)
    c2 = phi2.dot(b2)
    c4 =  phi2.dot(b4)

    #Step 3: pixel exchange
    r1 = re.R_matrix(phi1, lamb)
    r2 = re.R_matrix(phi2, lamb)
    #e1 is block1 after exchange
    e1, e2 = re.rand_pixel_exchange(c1, c2, r1)
    e2, e3 = re.rand_pixel_exchange(c2, c3, r2)
    e3, e4 = re.rand_pixel_exchange(c3, c4, r1)
    e4, e1 = re.rand_pixel_exchange(c4, c1, r2)

    #construct encrypted image
    return np.bmat([[e1, e2], [e3, e4]])

def decompress_decrypt():
    #Step 1: read into blocks 
    #img = cv2.imread(sys.argv[1])
    img = compress_encrypt()
    c1, c2, c3, c4 = read_into_blocks(img)

    #Step 2: inverse pixel exchange
    d4, d1 = re.rand_pixel_exchange(c4, c1, r2)
    d3, d4 = re.rand_pixel_exchange(c3, c4, r1)
    d2, d3 = re.rand_pixel_exchange(c2, c3, r2)
    d1, d2 = re.rand_pixel_exchange(c1, c2, r1)
    print("size of d4: ", d4.shape)
    print("phi1 size: ", phi1.shape)
    #Step 3: run SL0 algorithm with phi matrix
    sigma_min = 1
    alpha1 = sl.SL0(cv2.dct(phi1), d1.flatten(), sigma_min)
    alpha2 = sl.SL0(cv2.dct(phi2), d2.flatten(), sigma_min)
    alpha3 = sl.SL0(cv2.dct(phi1), d3.flatten(), sigma_min)
    alpha4 = sl.SL0(cv2.dct(phi2), d4.flatten(), sigma_min)
    print("size of alpha1: ", alpha1.shape)

    #convert back to space domain
    b1 = cv2.idct(alpha1)
    b2 = cv2.idct(alpha2)
    b3 = cv2.idct(alpha3)
    b4 = cv2.idct(alpha4)

    #Step 4: constrct the decripted image
    return np.bmat([[b1, b2] [b3, b4]])

if __name__ == '__main__':
    result = decompress_decrypt()
    print("size of result: ", result.shape)
