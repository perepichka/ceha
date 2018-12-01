"""
This module contains the entire process of compressed sensing as proposed
by the paper.

"""

import argparse
import cv2
import sys
import logging

from scipy import stats

from SL0 import *

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

    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    row_idx = img_height // 2
    col_idx = img_width // 2

    #Step 1: read and divide into blocks
    blocks = img[:row_idx, :col_idx], img[row_idx:, :col_idx], img[row_idx:, col_idx:], img[:row_idx, col_idx:]

    #Step 2: measure blocks with phi_matrix
    lamb = 2
    log_map1 = logistic_map(0.11, img_width//2)
    log_map2 = logistic_map(0.23, img_width//2)

    phi1 = phi_matrix(log_map1, lamb)
    phi2 = phi_matrix(log_map2, lamb)

    c1 = phi1.dot(blocks[0])
    c3 = phi1.dot(blocks[2])
    c2 = phi2.dot(blocks[1])
    c4 = phi2.dot(blocks[3])

    #Step 3: pixel exchange
    r1 = R_matrix(phi1, lamb)
    r2 = R_matrix(phi2, lamb)

    #e1 is block1 after exchange
    e1, e2 = rand_pixel_exchange(c1, c2, r1, mode='encrypt')
    e2, e3 = rand_pixel_exchange(e2, c3, r2, mode='encrypt')
    e3, e4 = rand_pixel_exchange(e3, c4, r1, mode='encrypt')
    e4, e1 = rand_pixel_exchange(e4, e1, r2, mode='encrypt')

    row_idx_e = e1.shape[0]
    col_idx_e = e1.shape[1]

    tst = np.zeros(shape=[e1.shape[0]*2, e1.shape[1]*2])
    tst[:row_idx_e, :col_idx_e], tst[row_idx_e:, :col_idx_e], tst[row_idx_e:, col_idx_e:], tst[:row_idx_e, col_idx_e:] = e1,e2,e3,e4

    #tst = cv2.normalize(tst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #tst = cv2.normalize(tst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #cv2.imshow('image', tst.astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #Step 2: inverse pixel exchange
    d4, d1 = rand_pixel_exchange(e4, e1, r2, mode='decrypt')
    d3, d4 = rand_pixel_exchange(e3, d4, r1, mode='decrypt')
    d2, d3 = rand_pixel_exchange(e2, d3, r2, mode='decrypt')
    d1, d2 = rand_pixel_exchange(d1, d2, r1, mode='decrypt')

    #Step 3: run SL0 algorithm with phi matrix

    phi1_dct = cv2.dct(phi1)
    phi2_dct = cv2.dct(phi2)

    d1_dct = cv2.dct(d1)
    d2_dct = cv2.dct(d2)
    d3_dct = cv2.dct(d3)
    d4_dct = cv2.dct(d4)

    s1 = SL0(phi1_dct, d1_dct, 1e-12)
    s2 = SL0(phi2_dct, d2_dct, 1e-12)
    s3 = SL0(phi1_dct, d3_dct, 1e-12)
    s4 = SL0(phi2_dct, d4_dct, 1e-12)

    print(s1.mean())
    print(s1.std())
    print(s1.min())

    f1 = cv2.idct(s1)
    f2 = cv2.idct(s2)
    f3 = cv2.idct(s3)
    f4 = cv2.idct(s4)

    final = np.zeros(shape=img.shape)
    final[:row_idx, :col_idx], final[row_idx:, :col_idx], final[row_idx:, col_idx:], final[:row_idx, col_idx:] = f1,f2,f3,f4

    final = cv2.normalize(final, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('image', final.astype(np.uint8))
    cv2.waitKey(0)

    #s1 = cv2.idct(



    #Step 4: constrct the decripted image
#
#if __name__ == '__main__':
#    result = read_into_blocks()
#    print("first block size: ", result[0].shape[:2][0], " ", result[0].shape[:2][1])
#    print("second block size: ", result[1].shape[:2][0], " ", result[1].shape[:2][1])
#    print("third block size: ", result[2].shape[:2][0], " ", result[2].shape[:2][1])
#    print("fourth block size: ", result[3].shape[:2][0], " ", result[3].shape[:2][1])
