import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

from src.wavelet_transform import generarte_haar_matrix, compress_image_haar_transform, decompress_image_haar_transform, haar_matrix_8X8

def load_images(folder):
    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))    
        if img is not None:
            images.append(img)
    
    return images


# Load Images
folder = "Images"
images = load_images(folder)

for num, img in enumerate(images):
    
    plt.title('img' + str(num))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
    compressed_img = img
    compressed_img = compress_image_haar_transform(compressed_img)
    cv2.imwrite('compressed_img' + str(num) + '.jpg', compressed_img)
    plt.title('compressed img' + str(num))
    plt.axis('off')
    plt.imshow(compressed_img)
    plt.show()

    decompressed_img = compressed_img
    decompressed_img = decompress_image_haar_transform(decompressed_img)
    cv2.imwrite('compressed_img' + str(num) + '.jpeg', compressed_img)
    cv2.imwrite('decompressed_img' + str(num) + '.jpeg', decompressed_img)
    plt.title('decompressed img' + str(num))
    plt.axis('off')
    plt.imshow(decompressed_img)
    plt.show()