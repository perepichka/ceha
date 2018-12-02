"""
This module contains the entire process of compressed sensing as proposed
by the paper.

"""

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

#from compress import compress, decompress
#from encrypt import encrypt, decrypt
from compress import *
from encrypt import *

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

DEFAULT_INPUT_IMAGE = '../data/input/lena-256x256.jpg'

DEFAULT_MODES = {
    'compress': 'Compresses an image',
    'encrypt': 'Encrypts an image',
    'decrypt' : 'Decrypts an image',
    'decompress' : 'Decompressed an image'
}

DEFAULT_HYPERPARAMS = {
    'x_01': 0.11,
    'x_02': 0.23,
    'mu': 3.99,
    'lambda': 2,
    'm_ratio': (3/4),
    'basis': 'dct',
    'method': 'l0'
}



class Process:
    """Main process class."""

    def __init__(self, input_img=None, output_img=None, modes=None, **kwargs):
        """Initializes the process object.

        :param str input_img: Input image.
        :param str output_img: Output image.
        :param list modes: List of modes to run.
        
        """

        if input_img is None or output_img is None:
            raise Exception('Error: Image paths not correctly specified.')


        self.input_img_path = input_img
        self.output_img_path = output_img

        # Necessary for callback functionality
        self._encrypt = encrypt
        self._decrypt = decrypt
        self._compress = compress
        self._decompress = decompress

    def read_image(self):
        """Reads an image file using opencv."""

        self.image = cv2.imread(input_img_path)
        img_height, img_width = img.shape[:2]

        logging.debug("height: {} , width: {} ".format(img_height, img_width))
        assert (img_height == img_width), "Image needs to be square!" 
        assert (img_height > 2 and img_width > 2), "Image is too small!" 

        self.img_height = img_height
        self.img_width = img_width

    def write_image(self):
        pass #@TODO Add image writing

    def split_image(self):
        """Reads image and splits it into blocks."""

        # Divide the array into 4 blocks, block 1 is at top left corner,
        # going counter-clockwise:
        # |--------|--------|
        # |        |        |
        # |   b1   |   b4   |
        # |        |        |
        # |--------|--------|
        # |        |        |
        # |   b2   |   b3   |
        # |        |        |
        # |--------|--------|

        row_idx = img_height // 2
        col_idx = img_width // 2

        b1 = img[:row_idx, :col_idx].copy()
        b2 = img[row_idx:, :col_idx].copy()
        b3 = img[row_idx:, col_idx:].copy()
        b4 = img[:row_idx, col_idx:].copy()

        return (b1,b2,b3,b4)

    def __call__(self):
        """Callback function controlling execution of the process."""

        for mode in self.modes:

            f = getattr(self, '_' + mode) 

            if f is None:
                logging.warning('Note, unkown mode "{}"'.format(mode))
                continue









if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compressed sensing.')

    parser.add_argument('--output',
        type=str, help='Input path to image to compress and encrypt.'
    )
    parser.add_argument(
        '--input', type=str, help='Input path to image to compress and encrypt.'
    )
    parser.add_argument(
        '--modes', type=str, nargs='+', default=DEFAULT_MODES,
        #help='Chose one or more of these modes:\n\n {}'.format(
        #    '\n'.join(' " ' + k + ' ":\t' + v for k,v in DEFAULT_MODES.items())
        #)
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
        '--lambda', type=float, default=DEFAULT_HYPERPARAMS.get('lambda'),
        help='Lambda for measuring matrix. Necessary if doing encryption or decryption'
    )
    parser.add_argument(
        '--mu', type=float, default=DEFAULT_HYPERPARAMS.get('mu'),
        help='Mu for logistic map. Necessary if doing encryption or decryption'
    )

    args = parser.parse_args()
    
    modes = []
    for m in args.modes:
        if m in DEFAULT_MODES.keys() and m not in modes:
            modes.append(m)


    #read image from command line
    img = cv2.imread(DEFAULT_INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    img_width, img_height = img.shape[:2]

    img = cv2.medianBlur(img,3)
    cv2.imshow('image orig', img.astype(np.uint8))
    cv2.waitKey(0)

    img = img.astype(np.float32)

    #print(img.mean())
    #print(img.std())
    #print(img.max())
    #print(img.min())

    #img = img.astype(np.int8)
    #img = cv2.normalize(img, None, alpha=-128, beta=128, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8S)
    #img = cv2.normalize(img, None, alpha=-1000000, beta=100000, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    #img = cv2.normalize(img, None, alpha=-1000000, beta=100000, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    #img = np.float32(img)#/255.0

    print('---------')
    print(img.mean())
    print(img.std())
    print(img.max())
    print(img.min())
    print('---------')

    #img = cv2.normalize(img, None, alpha=-128, beta=128, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imshow('image back', img_back.astype(np.uint8))
    #cv2.waitKey(0)

    row_idx = img_height // 2
    col_idx = img_width // 2

    #Step 1: read and divide into blocks
    blocks = img[:row_idx, :col_idx], img[row_idx:, :col_idx], img[row_idx:, col_idx:], img[:row_idx, col_idx:]

    #Step 2: measure blocks with phi_matrix
    lamb = 2
    log_map1 = logistic_map(0.11, img_width//2)
    log_map2 = logistic_map(0.23, img_width//2)
    #log_map1 = np.random.uniform(0,1,img_width//2)
    #log_map2 = np.random.uniform(0,1,img_width//2)

    m = int((3/4) * img_width//2)
    #m = int((1/4) * img_width//2)

    had_src = hadamard(img_width//2).astype(np.float32)

    phi1_indices = np.argsort(log_map1)
    phi2_indices = np.argsort(log_map2)

    phi1 = had_src[phi1_indices[:m],:]
    phi2 = had_src[phi2_indices[:m],:]

    phi1 = phi_matrix(log_map1, lamb)
    phi2 = phi_matrix(log_map2, lamb)
    #phi1 = phi_matrix(0.11, img_width//2, lamb)
    #phi2 = phi_matrix(0.23, img_width//2, lamb)
    #phi1 = np.random.rand(phi2.shape[0] * phi2.shape[1]).reshape(phi2.shape) * lamb
    #phi2 = np.random.rand(phi2.shape[0] * phi2.shape[1]).reshape(phi2.shape) * lamb

    b1,b2,b3,b4 = blocks[0],blocks[1],blocks[2],blocks[3]

    b1_dct = cv2.dct(blocks[0])
    b2_dct = cv2.dct(blocks[1])
    b3_dct = cv2.dct(blocks[2])
    b4_dct = cv2.dct(blocks[3])

    #print(np.count_nonzero(b1_dct))
    ##b1_dct[np.abs(b1_dct) < 0.01] = 0
    ##b2_dct[np.abs(b2_dct) < 0.01] = 0
    ##b3_dct[np.abs(b3_dct) < 0.01] = 0
    ##b3_dct[np.abs(b4_dct) < 0.01] = 0
    #print(np.count_nonzero(b1_dct))
    #b1 = cv2.idct(b1_dct)
    #b2 = cv2.idct(b2_dct)
    #b3 = cv2.idct(b3_dct)
    #b4 = cv2.idct(b4_dct)

    #img[:row_idx, :col_idx], img[row_idx:, :col_idx], img[row_idx:, col_idx:], img[:row_idx, col_idx:] = b1,b2,b3,b4
    #img = np.uint8(img*255.0)

    #c1 = phi1.dot(b1)
    #c3 = phi1.dot(b3)
    #c2 = phi2.dot(b2)
    #c4 = phi2.dot(b4)

    c1 = phi1.dot(cv2.dct(b1))
    c3 = phi1.dot(cv2.dct(b3))
    c2 = phi2.dot(cv2.dct(b2))
    c4 = phi2.dot(cv2.dct(b4))

    #Step 3: pixel exchange
    #r1 = R_matrix(phi1, lamb)
    #r2 = R_matrix(phi2, lamb)

    ##e1 is block1 after exchange
    #e1, e2 = rand_pixel_exchange(c1, c2, r1, mode='encrypt')
    #e2, e3 = rand_pixel_exchange(e2, c3, r2, mode='encrypt')
    #e3, e4 = rand_pixel_exchange(e3, c4, r1, mode='encrypt')
    #e4, e1 = rand_pixel_exchange(e4, e1, r2, mode='encrypt')

    ##Step 2: inverse pixel exchange
    #d4, d1 = rand_pixel_exchange(e4, e1, r2, mode='decrypt')
    #d3, d4 = rand_pixel_exchange(e3, d4, r1, mode='decrypt')
    #d2, d3 = rand_pixel_exchange(e2, d3, r2, mode='decrypt')
    #d1, d2 = rand_pixel_exchange(d1, d2, r1, mode='decrypt')

    d1=c1
    d2=c2
    d3=c3
    d4=c4

    #Step 3: run SL0 algorithm with phi matrix

    #phi1_dct = cv2.dct(phi1)
    #phi2_dct = cv2.dct(phi2)
    #phi1_dct = fftpack.dct(phi1, type=2, norm='ortho',axis=-1)
    #phi2_dct = fftpack.dct(phi2, type=2, norm='ortho',axis=-1)
    phi1_dct = phi1
    phi2_dct = phi2

    #d1_dct = cv2.dct(d1)
    #d2_dct = cv2.dct(d2)
    #d3_dct = cv2.dct(d3)
    #d4_dct = cv2.dct(d4)
    #d1_dct = fftpack.dct(d1, norm='ortho', axis=-1)
    #d2_dct = fftpack.dct(d2, norm='ortho', axis=-1)
    #d3_dct = fftpack.dct(d3, norm='ortho', axis=-1)
    #d4_dct = fftpack.dct(d4, norm='ortho', axis=-1)
    d1_dct = d1
    d2_dct = d2
    d3_dct = d3
    d4_dct = d4

    def tst_cost(s):
        s = s.reshape(b1.shape)

        # Does rounding
        #s[np.abs(s)<0.001] = 0.0
        #s[np.abs(s)<0.001] = 0.0
        #s_nrm = np.linalg.norm(s.flatten(), ord=0)

        #s = cv2.idct(s)
        dist = d1 - np.dot(phi1, s)
        #dist[np.abs(dist)<0.01] = 0.0
        #dist[np.abs(dist)<0.01] = 0.0

        #dist = d1 - np.dot(phi1, cv2.idct(s))
        
        diff_nrm = np.linalg.norm(dist, ord=2)

        return diff_nrm

    phi1_inv = np.linalg.pinv(phi1)
    init_s = np.dot(phi1_inv, d1).flatten()
    #init_s = cv2.dct(init_s)
    #phi1_inv = np.linalg.pinv(cv2.dct(phi1))
    #init_s = np.dot(phi1_inv, d1_dct)
    #init_s = cv2.idct(init_s)
    #phi1_inv = np.linalg.pinv(fftpack.dct(phi1, axis=-1))
    #init_s = np.dot(phi1_inv, d1)
    #init_s = fftpack.idct(init_s, axis=0)


    #tst = minimize(tst_cost, np.zeros(b1.shape), method='Newton-CG')

    bnds = [(-2, 257) for i in range(init_s.shape[0])]

    #tst = minimize(tst_cost, init_s, method='L-BFGS-B', options={'disp':True, 'maxiter':30, 'maxfun':30, 'maxls':20})
    #tst = minimize(tst_cost, init_s, method='L-BFGS-B', options={'disp':True, 'ftol':1.0})
    #tst = minimize(tst_cost, init_s, bounds=(0, 255), method='L-BFGS-B', options={'maxiter':200})
    #tst = minimize(tst_cost, np.zeros(b1.shape), method='Nelder-Mead', options={'maxiter':200})
    #print(tst.nit)
    #print(tst.fun)
    #print(tst.x)

    
    s1 = SL0(phi1_dct, d1_dct, 1e-12, sigma_decrease_factor=0.5, L=3)#true_s=cv2.dct(blocks[0]))
    s2 = SL0(phi2_dct, d2_dct, 1e-12, sigma_decrease_factor=0.5, L=3)#true_s=cv2.dct(blocks[1]))
    s3 = SL0(phi1_dct, d3_dct, 1e-12, sigma_decrease_factor=0.5, L=3)#true_s=cv2.dct(blocks[2]))
    s4 = SL0(phi2_dct, d4_dct, 1e-12, sigma_decrease_factor=0.5, L=3)#true_s=cv2.dct(blocks[3]))


    f1 = cv2.idct(s1)
    f2 = cv2.idct(s2)
    f3 = cv2.idct(s3)
    f4 = cv2.idct(s4)

    #f1 = cv2.idct(tst.x.reshape(f1.shape))
    #f1 = tst.x.reshape(f1.shape)
    #f1 = fftpack.idct(s1, type=2, norm='ortho', axis=0)
    #f2 = fftpack.idct(s2, type=2, norm='ortho', axis=0)
    #f3 = fftpack.idct(s3, type=2, norm='ortho', axis=0)
    #f4 = fftpack.idct(s4, type=2, norm='ortho', axis=0)
    #f1 = s1
    #f2 = s2
    #f3 = s3
    #f4 = s4


    final = np.zeros(shape=img.shape)
    final[:row_idx, :col_idx], final[row_idx:, :col_idx], final[row_idx:, col_idx:], final[:row_idx, col_idx:] = f1,f2,f3,f4

    tsts = cv2.normalize(final, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('image final', tsts.astype(np.uint8))
    cv2.waitKey(0)
    
    final[final>255] = 255
    final[final<0] = 0

    cv2.imshow('image final 2', final.astype(np.uint8))
    cv2.waitKey(0)

    #print('---------')
    #print(final.mean())
    #print(final.std())
    #print(final.max())
    #print(final.min())
    #print('--------- diff -------- ')
    #print(np.abs(final-img).mean())
    #print(np.abs(final-img).std())
    #print(np.abs(final-img).max())
    #print(np.abs(final-img).min())





    #Step 4: constrct the decripted image
#
#if __name__ == '__main__':
#    result = read_into_blocks()
#    print("first block size: ", result[0].shape[:2][0], " ", result[0].shape[:2][1])
#    print("second block size: ", result[1].shape[:2][0], " ", result[1].shape[:2][1])
#    print("third block size: ", result[2].shape[:2][0], " ", result[2].shape[:2][1])
#    print("fourth block size: ", result[3].shape[:2][0], " ", result[3].shape[:2][1])
