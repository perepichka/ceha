"""
This module contains the entire process of compressed sensing as proposed
by the paper.

"""

import argparse
import cv2
import sys
import logging

from compress import compress, decompress
from encrypt import encrypt, decrypt

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

DEFAULT_INPUT_IMAGE = '../data/input/lena256x265.jpg'

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

    def __init__(self, input_img=None, output_img=None, modes, **kwargs):
        """Initializes the process object.

        :param input_img str: Input image.
        :param output_img str: Output image.
        :param modes list: List of modes to run.
        
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
        '--modes', type=str, nargs='+',
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
        '--lambda', type=, default=DEFAULT_HYPERPARAMS.get('lambda'),
        help='Lambda for measuring matrix. Necessary if doing encryption or decryption'
    )
    parser.add_argument(
        '--mu', type=, default=DEFAULT_HYPERPARAMS.get('mu'),
        help='Mu for logistic map. Necessary if doing encryption or decryption'
    )

    args = parser.parse_args()
    
    modes = []
    for m in args.modes:
        if m in DEFAULT_MODES.keys() and m not in modes:
            modes.append(m)

    #result = read_into_blocks()
    #print("first block size: ", result[0].shape[:2][0], " ", result[0].shape[:2][1])
    #print("second block size: ", result[1].shape[:2][0], " ", result[1].shape[:2][1])
    #print("third block size: ", result[2].shape[:2][0], " ", result[2].shape[:2][1])
    #print("fourth block size: ", result[3].shape[:2][0], " ", result[3].shape[:2][1])

#def compress_encrypt():
#    """
#    compress and encrypt input image
#    """
#
#    #read image from command line
#    img = cv2.imread(sys.argv[1])
#    img_width, img_height = img.shape[:2]
#
#    #Step 1: read and divide into blocks
#    blocks = read_into_blocks(img)
#
#    #Step 2: measure blocks with phi_matrix
#    lamb = 2
#    log_map1 = cp.logistic_map(0.11, img_width)
#    log_map2 = cp.logistic_map(0.23, img_width)
#    phi1 = cp.phi_matrix(log_map1, lamb)
#    phi2 = cp.phi_matrix(log_map2, lamb)
#    c1 = phi1.dot(blocks[0])
#    c3 = phi1.dot(blocks[2])
#    c2 = phi2.dot(blocks[1])
#    c4 =  phi2.dot(blocks[3])
#
#    #Step 3: pixel exchange
#    r1 = R_matrix(phi1, lamb)
#    r2 = R_matrix(phi2, lamb)
#    #e1 is block1 after exchange
#    e1, e2 = re.rand_pixel_exchange(c1, c2, r1)
#    e2, e3 = re.rand_pixel_exchange(c2, c3, r2)
#    e3, e4 = re.rand_pixel_exchange(c3, c4, r1)
#    e4, e1 = re.rand_pixel_exchange(c4, c1, r2)
#
#    #construct encrypted image
#    return np.bmat([[e1, e2], [e3, e4]])
#
#def decompress_decrypt():
#    #Step 1: read into blocks 
#    img = cv2.imread(sys.argv[1])
#    c1, c2, c3, c4 = read_into_blocks(img)
#
#    #Step 2: inverse pixel exchange
#    d4, d1 = re.rand_pixel_exchange(c4, c1, r2)
#    d3, d4 = re.rand_pixel_exchange(c3, c4, r1)
#    d2, d3 = re.rand_pixel_exchange(c2, c3, r2)
#    d1, d2 = re.rand_pixel_exchange(c1, c2, r1)
#
#    #Step 3: run SL0 algorithm with phi matrix
#
#
#    #Step 4: constrct the decripted image
#
#if __name__ == '__main__':
#    result = read_into_blocks()
#    print("first block size: ", result[0].shape[:2][0], " ", result[0].shape[:2][1])
#    print("second block size: ", result[1].shape[:2][0], " ", result[1].shape[:2][1])
#    print("third block size: ", result[2].shape[:2][0], " ", result[2].shape[:2][1])
#    print("fourth block size: ", result[3].shape[:2][0], " ", result[3].shape[:2][1])
