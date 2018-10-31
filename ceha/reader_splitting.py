import cv2
import numpy as np

inputImage = cv2.imread('lena.png')

height, width, channels = inputImage.shape

if width != height:
    print("This is not a square image")
else:

    # Initialize four splitted matrices

    splitted_matrix_length = int(width / 2)

    splitted_matrix1 = np.zeros((splitted_matrix_length, splitted_matrix_length, 3), dtype=np.uint8)
    splitted_matrix2 = np.zeros((splitted_matrix_length, splitted_matrix_length, 3), dtype=np.uint8)
    splitted_matrix3 = np.zeros((splitted_matrix_length, splitted_matrix_length, 3), dtype=np.uint8)
    splitted_matrix4 = np.zeros((splitted_matrix_length, splitted_matrix_length, 3), dtype=np.uint8)

    for i in range(splitted_matrix_length):
        for j in range(splitted_matrix_length):
            splitted_matrix1[i][j] = inputImage[i][j]

    for i in range(splitted_matrix_length):
        for j in range(splitted_matrix_length):
            splitted_matrix2[i][j] = inputImage[i][splitted_matrix_length + j]

    for i in range(splitted_matrix_length):
        for j in range(splitted_matrix_length):
            splitted_matrix3[i][j] = inputImage[splitted_matrix_length + i][j]

    for i in range(splitted_matrix_length):
        for j in range(splitted_matrix_length):
            splitted_matrix4[i][j] = inputImage[splitted_matrix_length + i][splitted_matrix_length + j]

    print(inputImage[0][0])
    print(inputImage[0][1])
    print(inputImage[1][0])

    cv2.imshow('1', splitted_matrix1)
    cv2.imshow('2', splitted_matrix2)
    cv2.imshow('3', splitted_matrix3)
    cv2.imshow('4', splitted_matrix4)
    cv2.waitKey()
