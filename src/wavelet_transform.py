import numpy as np
import cv2

def one_dim_array_haar_transform(array):
    length = len(array)
    approximation_coefficients = []
    detail_coefficients = []

    # Check length is odd or not
    if( length%2 == 1):
        print("ODD length, last element is ignored")
        length = length - 1

    i = 0
    while i < length:
        avg = array[i] + array[i+1]
        avg = avg/2
        approximation_coefficients.append(avg)

        diff = array[i] - array[i+1]
        diff = diff/2
        detail_coefficients.append(diff)

        i = i + 2
    
    return approximation_coefficients + detail_coefficients


def generarte_haar_matrix(n, normalized=False):
    #Source: 
    # 0. https://en.wikipedia.org/wiki/Haar_wavelet
    # 1. http://fourier.eng.hmc.edu/e161/lectures/Haar/index.html
    # 2. https://docs.scipy.org/doc/numpy/reference/generated/numpy.kron.html

    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = generarte_haar_matrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h

def compress_image_haar_transform_A(img):

    if(img.shape[0] != img.shape[1]):
        print("width_image != height_image")
        return None

    n = img.shape[0]
    haar_matrix = generarte_haar_matrix(n)
    transpose_haar_matrix = haar_matrix.transpose()

    channels = img.shape[2]

    for i in range(channels):
        I = img[:,:,i] 
        img[:,:,i] = ( transpose_haar_matrix.dot(I) ).dot( haar_matrix )
    
    return img

def decompress_image_haar_transform_A(img):

    if(img.shape[0] != img.shape[1]):
        print("width_image != height_image")
        return None

    n = img.shape[0]
    haar_matrix = generarte_haar_matrix(n) 
    transpose_haar_matrix = haar_matrix.transpose()

    inverse_haar_matrix = np.linalg.inv( haar_matrix )
    inverse_transpose_haar_matrix = np.linalg.inv( transpose_haar_matrix )

    channels = img.shape[3]

    for i in range(channels):
        I = img[:,:,i] 
        img[:,:,i] = ( inverse_transpose_haar_matrix.dot(I) ).dot( inverse_haar_matrix )
    
    return img

def haar_matrix_8X8():
    a = [1/8, 1/8, 1/4, 0, 1/2, 0, 0, 0]
    b = [1/8, 1/8, 1/4, 0, -1/2, 0, 0, 0]
    c = [1/8, 1/8, -1/4, 0, 0, 1/2, 0, 0]
    d = [1/8, 1/8, -1/4, 0, 0, -1/2, 0, 0]
    e = [1/8, -1/8, 0, 1/4, 0, 0, 1/2, 0]
    f = [1/8, -1/8, 0, -1/4, 0, 0, -1/2, 0]
    g = [1/8, -1/8, 0, -1/4, 0, 0, 0, 1/2]
    h = [1/8, -1/8, 0, -1/4, 0, 0, 0, -1/2]

    haar_matrix = np.array([ a, b, c, d, e ,f ,g, h])

    return haar_matrix

def compress_image_haar_transform(img):

    haar_matrix = haar_matrix_8X8()
    transpose_haar_matrix = haar_matrix.transpose()

    for channels in range(img.shape[2]):
        right = 8
        bottom = 8

        while right <= img.shape[0]:
            while bottom <= img.shape[1]:
                I = img[right-8: right, bottom-8: bottom, channels]
                img[right-8: right, bottom-8: bottom, channels] = ( transpose_haar_matrix.dot(I) ).dot( haar_matrix )
                bottom = bottom + 8
            
            right = right + 8

    return img

def decompress_image_haar_transform(img):

    haar_matrix = haar_matrix_8X8()
    transpose_haar_matrix = haar_matrix.transpose()

    inverse_haar_matrix = np.linalg.inv( haar_matrix )
    inverse_transpose_haar_matrix = np.linalg.inv( transpose_haar_matrix )

    for channels in range(img.shape[2]):
        right = 8
        bottom = 8

        while right <= img.shape[0]:
            while bottom <= img.shape[1]:
                I = img[right-8: right, bottom-8: bottom, channels]
                img[right-8: right, bottom-8: bottom, channels] = ( inverse_transpose_haar_matrix.dot(I) ).dot( inverse_haar_matrix )
                bottom = bottom + 8
            
            right = right + 8

    return img




