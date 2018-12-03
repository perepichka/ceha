"""
This module contains the entire process of compressed sensing as proposed
by the paper.

"""

import os
import argparse
import cv2
import sys
import logging

from scipy import stats
from scipy import fftpack
from scipy.optimize import minimize
from scipy.linalg import hadamard

from SL0 import *
from psnr import psnr

from compress import *
from encrypt import *

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

DEFAULT_INPUT_IMAGE = '../data/input/lena-256x256.jpg'

DEFAULT_MODES = {
    'compress': 'Compresses and encrypts an image',
    'decompress': 'Decompresses and dencrypts an image',
    #'compress': 'Compresses an image',
    #'encrypt': 'Encrypts an image',
    #'decrypt' : 'Decrypts an image',
    #'decompress' : 'Decompressed an image'
}

DEFAULT_DISPLAY = True

DEFAULT_HYPERPARAMS = {
    'x_01': 0.11,
    'x_02': 0.23,
    'mu': 3.99,
    'lamb': 2,
    'm_ratio': (3/4),
    'basis': 'dct',
    'method': 'l0'
}

DEFAULT_METHOD = 'circulant'

def compress(img, x_01, x_02, mu, lamb, m_ratio, display, method, output):

    img_width, img_height = img.shape[:2]

    #img = cv2.medianBlur(img,3)

    img = img.astype(np.float32)

    row_idx = img_height // 2
    col_idx = img_width // 2

    #Step 1: read and divide into blocks
    b1,b2,b3,b4 = img[:row_idx, :col_idx], img[row_idx:, :col_idx], img[row_idx:, col_idx:], img[:row_idx, col_idx:]

    #Step 2: measure blocks with phi_matrix
    log_map1 = logistic_map(0.11, img_width//2)
    log_map2 = logistic_map(0.23, img_width//2)

    if method == 'hadamard':
        m = int(m_ratio * img_width//2)
        had_src = hadamard(img_width//2).astype(np.float32)
        phi1_indices = np.argsort(log_map1)
        phi2_indices = np.argsort(log_map2)

        phi1 = had_src[phi1_indices[:m],:]
        phi2 = had_src[phi2_indices[:m],:]
    elif method == 'circulant':
        phi1 = phi_matrix(log_map1, lamb, m_ratio)
        phi2 = phi_matrix(log_map2, lamb, m_ratio)

    c1 = phi1.dot(cv2.dct(b1))
    c3 = phi1.dot(cv2.dct(b3))
    c2 = phi2.dot(cv2.dct(b2))
    c4 = phi2.dot(cv2.dct(b4))

    # saves compressed
    c_combine = np.block([[c1, c4], [c2, c3]])

    c_name = os.path.splitext(os.path.basename(output))[0]
    c_file = output.replace(c_name, c_name + '_compressed')
    cv2.imwrite(c_file, c_combine.astype(np.uint8))

    #Step 3: pixel exchange
    r1 = R_matrix(phi1, lamb)
    r2 = R_matrix(phi2, lamb)

    #e1 is block1 after exchange
    e1, e2 = rand_pixel_exchange(c1, c2, r1, mode='encrypt')
    e2, e3 = rand_pixel_exchange(e2, c3, r2, mode='encrypt')
    e3, e4 = rand_pixel_exchange(e3, c4, r1, mode='encrypt')
    e4, e1 = rand_pixel_exchange(e4, e1, r2, mode='encrypt')

    encrypted = np.block([[e1, e4], [e2, e3]])

    if display:
        cv2.imshow('image encrypted', encrypted.astype(np.uint8))
        cv2.waitKey(0)

    e_name = os.path.splitext(os.path.basename(output))[0]
    e_file = output.replace(e_name, e_name + '_encrypted')
    cv2.imwrite(e_file, encrypted.astype(np.uint8))

    return encrypted

def decompress(img, x_01, x_02, mu, lamb, m_ratio, display, method, output):


    img_width, img_height = img.shape[1], img.shape[1]
    row_idx = img_height // 2
    col_idx = img_width // 2

    # Calculate our phi and stuff
    log_map1 = logistic_map(0.11, img_width//2)
    log_map2 = logistic_map(0.23, img_width//2)

    if method == 'hadamard':
        m = int(m_ratio * img_width//2)
        had_src = hadamard(img_width//2).astype(np.float32)
        phi1_indices = np.argsort(log_map1)
        phi2_indices = np.argsort(log_map2)

        phi1 = had_src[phi1_indices[:m],:]
        phi2 = had_src[phi2_indices[:m],:]
    elif method == 'circulant':
        phi1 = phi_matrix(log_map1, lamb, m_ratio)
        phi2 = phi_matrix(log_map2, lamb, m_ratio)

    r1 = R_matrix(phi1, lamb)
    r2 = R_matrix(phi2, lamb)

    img_width_e, img_height_e = img.shape[:2]
    row_idx_e =  img_width_e // 2
    col_idx_e = img_height_e // 2

    e1,e2,e3,e4 = img[:row_idx_e, :col_idx], img[row_idx_e:, :col_idx], img[row_idx_e:, col_idx:], img[:row_idx_e, col_idx:]

    #Step 2: inverse pixel exchange
    d4, d1 = rand_pixel_exchange(e4, e1, r2, mode='decrypt')
    d3, d4 = rand_pixel_exchange(e3, d4, r1, mode='decrypt')
    d2, d3 = rand_pixel_exchange(e2, d3, r2, mode='decrypt')
    d1, d2 = rand_pixel_exchange(d1, d2, r1, mode='decrypt')

    #Step 3: run SL0 algorithm with phi matrix
    s1 = SL0(phi1, d1, 1e-12, sigma_decrease_factor=0.5, L=3)
    s2 = SL0(phi2, d2, 1e-12, sigma_decrease_factor=0.5, L=3)
    s3 = SL0(phi1, d3, 1e-12, sigma_decrease_factor=0.5, L=3)
    s4 = SL0(phi2, d4, 1e-12, sigma_decrease_factor=0.5, L=3)

    f1 = cv2.idct(s1)
    f2 = cv2.idct(s2)
    f3 = cv2.idct(s3)
    f4 = cv2.idct(s4)

    final = np.zeros(shape=img.shape)
    final = np.block([[f1, f4], [f2, f3]])

    #tsts = cv2.normalize(final, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imshow('image final', tsts.astype(np.uint8))
    #cv2.waitKey(0)
    
    final[final>255] = 255
    final[final<0] = 0

    f_name = os.path.splitext(os.path.basename(output))[0]
    f_file = output.replace(f_name, f_name + '_decompressed')
    cv2.imwrite(f_file,final.astype(np.uint8))

    if display:
        cv2.imshow('image final', final.astype(np.uint8))
        cv2.waitKey(0)

    return final

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compressed sensing.')

    parser.add_argument('--output',
        type=str, help='Input path to image to compress and encrypt.'
    )
    parser.add_argument(
        '--input', type=str, help='Input path to image to compress and encrypt.',
        default=DEFAULT_INPUT_IMAGE
    )
    parser.add_argument(
        '--method', type=str, default=DEFAULT_METHOD, 
        help='Choose which method to use: circulant matrix or hadamard'
    )
    parser.add_argument(
        '--mode', type=str, nargs='+', default=DEFAULT_MODES,
        #help='Chose one or more of these modes:\n\n {}'.format(
        #    '\n'.join(' " ' + k + ' ":\t' + v for k,v in DEFAULT_MODES.items())
        #)
    )
    parser.add_argument(
        '--m_ratio', type=float, default=DEFAULT_HYPERPARAMS.get('m_ratio'),
        help='Compression ratio multiplier.'
    )
    parser.add_argument(
        '--x_01', type=float, default=DEFAULT_HYPERPARAMS.get('x_01'),
        help='Image key x_01. Necessary if doing encryption or decryption'
    )
    parser.add_argument(
        '--x_02', type=float, default=DEFAULT_HYPERPARAMS.get('x_02'),
        help='Image key x_02. Necessary if doing encryption or decryption'
    )
    parser.add_argument(
        '--lamb', type=float, default=DEFAULT_HYPERPARAMS.get('lamb'),
        help='Lambda for measuring matrix. Necessary if doing encryption or decryption'
    )
    parser.add_argument(
        '--mu', type=float, default=DEFAULT_HYPERPARAMS.get('mu'),
        help='Mu for logistic map. Necessary if doing encryption or decryption'
    )
    parser.add_argument(
        '--display', type=bool, default=DEFAULT_DISPLAY,
        help='Whether or not to display results.'
    )

    args = parser.parse_args()
    
    mode = []
    for m in args.mode:
        if m in DEFAULT_MODES.keys() and m not in mode:
            mode.append(m)

    # If no modes specified, to the whole pipeline
    if len(mode) == 0:
        mode.append('compress')
        mode.append('decompress')

    # Sets up stuff we'll need
    m_ratio = args.m_ratio
    x_01 = args.x_01
    x_02 = args.x_01
    mu = args.mu
    lamb = args.lamb
    display = args.display
    method = args.method
    output = args.output

    if output is None:
        output = args.input.replace('input', 'output')
    
    # Reads the image
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    if display:
        cv2.imshow('image orig', img.astype(np.uint8))
        cv2.waitKey(0)

    orig_img = None

    if 'compress' in mode:

        orig_img = img.copy()
        res_img = compress(img, x_01, x_01, mu, lamb, m_ratio, display, method, output)


        img = res_img

    if 'decompress' in mode:

        res_img = decompress(img, x_01, x_01, mu, lamb, m_ratio, display, method, output)

        if orig_img is not None:
            psnr_val = psnr(orig_img, res_img)
            print(psnr_val)
            p_name = os.path.splitext(os.path.basename(output))[0]
            p_file = output.replace(p_name, p_name + 'psnr')
            p_file = output.replace(os.path.splitext(os.path.basename(output))[1], '.txt')
            with open(p_file, 'w') as f:
                f.write(str(psnr_val))

##    img_width, img_height = img.shape[:2]
##
##    img = cv2.medianBlur(img,3)
##    cv2.imshow('image orig', img.astype(np.uint8))
##    cv2.waitKey(0)
##
##    img = img.astype(np.float32)
##
##    row_idx = img_height // 2
##    col_idx = img_width // 2
##
##    #Step 1: read and divide into blocks
##    b1,b2,b3,b4 = img[:row_idx, :col_idx], img[row_idx:, :col_idx], img[row_idx:, col_idx:], img[:row_idx, col_idx:]
##
##    #Step 2: measure blocks with phi_matrix
##    lamb = 2
##    log_map1 = logistic_map(0.11, img_width//2)
##    log_map2 = logistic_map(0.23, img_width//2)
##    #log_map1 = np.random.uniform(0,1,img_width//2)
##    #log_map2 = np.random.uniform(0,1,img_width//2)
##
##    m = int((3/4) * img_width//2)
##
##    had_src = hadamard(img_width//2).astype(np.float32)
##
##    phi1_indices = np.argsort(log_map1)
##    phi2_indices = np.argsort(log_map2)
##
##    phi1 = had_src[phi1_indices[:m],:]
##    phi2 = had_src[phi2_indices[:m],:]
##
##    phi1 = phi_matrix(log_map1, lamb)
##    phi2 = phi_matrix(log_map2, lamb)
##
##    b1_dct = cv2.dct(b1)
##    b2_dct = cv2.dct(b2)
##    b3_dct = cv2.dct(b3)
##    b4_dct = cv2.dct(b4)
##
##    c1 = phi1.dot(cv2.dct(b1))
##    c3 = phi1.dot(cv2.dct(b3))
##    c2 = phi2.dot(cv2.dct(b2))
##    c4 = phi2.dot(cv2.dct(b4))
##
##    #Step 3: pixel exchange
##    r1 = R_matrix(phi1, lamb)
##    r2 = R_matrix(phi2, lamb)
##
##    #e1 is block1 after exchange
##    e1, e2 = rand_pixel_exchange(c1, c2, r1, mode='encrypt')
##    e2, e3 = rand_pixel_exchange(e2, c3, r2, mode='encrypt')
##    e3, e4 = rand_pixel_exchange(e3, c4, r1, mode='encrypt')
##    e4, e1 = rand_pixel_exchange(e4, e1, r2, mode='encrypt')
##
##    #Step 2: inverse pixel exchange
##    d4, d1 = rand_pixel_exchange(e4, e1, r2, mode='decrypt')
##    d3, d4 = rand_pixel_exchange(e3, d4, r1, mode='decrypt')
##    d2, d3 = rand_pixel_exchange(e2, d3, r2, mode='decrypt')
##    d1, d2 = rand_pixel_exchange(d1, d2, r1, mode='decrypt')
##
##    #Step 3: run SL0 algorithm with phi matrix
##
##    #phi1_dct = cv2.dct(phi1)
##    #phi2_dct = cv2.dct(phi2)
##    #phi1_dct = fftpack.dct(phi1, type=2, norm='ortho',axis=-1)
##    #phi2_dct = fftpack.dct(phi2, type=2, norm='ortho',axis=-1)
##    phi1_dct = phi1
##    phi2_dct = phi2
##
##    #d1_dct = cv2.dct(d1)
##    #d2_dct = cv2.dct(d2)
##    #d3_dct = cv2.dct(d3)
##    #d4_dct = cv2.dct(d4)
##    #d1_dct = fftpack.dct(d1, norm='ortho', axis=-1)
##    #d2_dct = fftpack.dct(d2, norm='ortho', axis=-1)
##    #d3_dct = fftpack.dct(d3, norm='ortho', axis=-1)
##    #d4_dct = fftpack.dct(d4, norm='ortho', axis=-1)
##    d1_dct = d1
##    d2_dct = d2
##    d3_dct = d3
##    d4_dct = d4
##
##    def tst_cost(s):
##        s = s.reshape(b1.shape)
##
##        # Does rounding
##        #s[np.abs(s)<0.001] = 0.0
##        #s[np.abs(s)<0.001] = 0.0
##        #s_nrm = np.linalg.norm(s.flatten(), ord=0)
##
##        #s = cv2.idct(s)
##        dist = d1 - np.dot(phi1, s)
##        #dist[np.abs(dist)<0.01] = 0.0
##        #dist[np.abs(dist)<0.01] = 0.0
##
##        #dist = d1 - np.dot(phi1, cv2.idct(s))
##        
##        diff_nrm = np.linalg.norm(dist, ord=2)
##
##        return diff_nrm
##
##    phi1_inv = np.linalg.pinv(phi1)
##    init_s = np.dot(phi1_inv, d1).flatten()
##    #init_s = cv2.dct(init_s)
##    #phi1_inv = np.linalg.pinv(cv2.dct(phi1))
##    #init_s = np.dot(phi1_inv, d1_dct)
##    #init_s = cv2.idct(init_s)
##    #phi1_inv = np.linalg.pinv(fftpack.dct(phi1, axis=-1))
##    #init_s = np.dot(phi1_inv, d1)
##    #init_s = fftpack.idct(init_s, axis=0)
##
##
##    #tst = minimize(tst_cost, np.zeros(b1.shape), method='Newton-CG')
##
##    bnds = [(-2, 257) for i in range(init_s.shape[0])]
##
##    #tst = minimize(tst_cost, init_s, method='L-BFGS-B', options={'disp':True, 'maxiter':30, 'maxfun':30, 'maxls':20})
##    #tst = minimize(tst_cost, init_s, method='L-BFGS-B', options={'disp':True, 'ftol':1.0})
##    #tst = minimize(tst_cost, init_s, bounds=(0, 255), method='L-BFGS-B', options={'maxiter':200})
##    #tst = minimize(tst_cost, np.zeros(b1.shape), method='Nelder-Mead', options={'maxiter':200})
##    #print(tst.nit)
##    #print(tst.fun)
##    #print(tst.x)
##
##    
##    s1 = SL0(phi1_dct, d1_dct, 1e-12, sigma_decrease_factor=0.5, L=3)
##    s2 = SL0(phi2_dct, d2_dct, 1e-12, sigma_decrease_factor=0.5, L=3)
##    s3 = SL0(phi1_dct, d3_dct, 1e-12, sigma_decrease_factor=0.5, L=3)
##    s4 = SL0(phi2_dct, d4_dct, 1e-12, sigma_decrease_factor=0.5, L=3)
##
##
##    f1 = cv2.idct(s1)
##    f2 = cv2.idct(s2)
##    f3 = cv2.idct(s3)
##    f4 = cv2.idct(s4)
##
##    #f1 = cv2.idct(tst.x.reshape(f1.shape))
##    #f1 = tst.x.reshape(f1.shape)
##    #f1 = fftpack.idct(s1, type=2, norm='ortho', axis=0)
##    #f2 = fftpack.idct(s2, type=2, norm='ortho', axis=0)
##    #f3 = fftpack.idct(s3, type=2, norm='ortho', axis=0)
##    #f4 = fftpack.idct(s4, type=2, norm='ortho', axis=0)
##    #f1 = s1
##    #f2 = s2
##    #f3 = s3
##    #f4 = s4
##
##
##    final = np.zeros(shape=img.shape)
##    final[:row_idx, :col_idx], final[row_idx:, :col_idx], final[row_idx:, col_idx:], final[:row_idx, col_idx:] = f1,f2,f3,f4
##
##    tsts = cv2.normalize(final, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
##    cv2.imshow('image final', tsts.astype(np.uint8))
##    cv2.waitKey(0)
##    
##    final[final>255] = 255
##    final[final<0] = 0
##
##    cv2.imshow('image final 2', final.astype(np.uint8))
##    cv2.waitKey(0)
##
##    #print('---------')
##    #print(final.mean())
##    #print(final.std())
##    #print(final.max())
##    #print(final.min())
##    #print('--------- diff -------- ')
##    #print(np.abs(final-img).mean())
##    #print(np.abs(final-img).std())
##    #print(np.abs(final-img).max())
##    #print(np.abs(final-img).min())
##
##
##
##
##
##    #Step 4: constrct the decripted image
###
###if __name__ == '__main__':
###    result = read_into_blocks()
###    print("first block size: ", result[0].shape[:2][0], " ", result[0].shape[:2][1])
###    print("second block size: ", result[1].shape[:2][0], " ", result[1].shape[:2][1])
###    print("third block size: ", result[2].shape[:2][0], " ", result[2].shape[:2][1])
###    print("fourth block size: ", result[3].shape[:2][0], " ", result[3].shape[:2][1])
