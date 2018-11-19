"""
This module contains the entire process of compressed sensing as proposed
by the paper
"""

import cv2
import sys
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

def read_into_blocks():
	#read image from command line and convert to a matrix
	img = cv2.imread(sys.argv[1])
	img_height, img_width = img.shape[:2]
	print("height: ", img_height," width: ", img_width)
	#divide the array into 4 blocks, block 1 is at top left corner, going counter-clockwise
	row_idx = img_height//2
	col_idx = img_width//2
	b1 = img[:row_idx, :col_idx].copy()
	b2 = img[row_idx:, :col_idx].copy()
	b3 = img[row_idx:, col_idx:].copy()
	b4 = img[:row_idx, col_idx:].copy()
	return (b1,b2,b3,b4)

if __name__ == '__main__':
	result = read_into_blocks()
	print("first block size: ", result[0].shape[:2][0], " ", result[0].shape[:2][1])
	print("second block size: ", result[1].shape[:2][0], " ", result[1].shape[:2][1])
	print("third block size: ", result[2].shape[:2][0], " ", result[2].shape[:2][1])
	print("fourth block size: ", result[3].shape[:2][0], " ", result[3].shape[:2][1])