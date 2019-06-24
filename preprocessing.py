import os
import random
import shutil
import matplotlib.pyplot as plt
from os import path
import numpy as np

import cv2


def parse_number_dataset():
    file_path = r"..\numbers-master"
    final_path = r"..\numbers"

    file_dict = {str(i): 0 for i in range(10)}

    for directory in os.listdir(file_path):
        dir_path = path.join(file_path, directory)
        for image_dir in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, image_dir)):
                image_path = path.join(dir_path, image_dir)
                for image_file in os.listdir(image_path):
                    current_image_path = os.path.join(image_path, image_file)
                    if os.path.isfile(current_image_path):
                        final_file_path = os.path.join(final_path, image_dir)
                        if not path.exists(final_file_path):
                            os.mkdir(final_file_path)
                        final_image_name = str(file_dict[image_dir]) + '.png'
                        file_dict[image_dir] += 1
                        shutil.copy(current_image_path, final_file_path)
                        os.rename(path.join(final_file_path, image_file), path.join(final_file_path, final_image_name))


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def merge_grid(image_number, grid_number):
    file_path = r"C:\Users\A\Desktop\PiRO\numbers\8"
    grid_path = r"C:\Users\A\Desktop\PiRO\PiRO_OCR\kratki_extracted"
    image_number = r'\{}.png'.format(image_number)
    grid_number = r'\{}.jpg'.format(grid_number)
    img = cv2.imread(file_path + image_number, 0)
    img = cv2.resize(img, (48, 48))
    img = img[:, 8:40]
    #print(img.shape)

    # plt.imshow(img, cmap='gray')
    # plt.show()

    img = adjust_gamma(img, gamma=0.4)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img = cv2.GaussianBlur(img, (3, 3), 2)
    img = adjust_gamma(img, gamma=0.4)

    # plt.imshow(img, cmap='gray')
    # plt.show()

    # _, img = cv2.threshold(img, 250, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # gaussian_kernel  =cv2.getGaussianKernel()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # img = cv2.erode(img, np.ones((3, 3), dtype=np.uint8))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    grid = cv2.imread(grid_path + grid_number, 0)
    grid = adjust_gamma(grid, gamma=0.2)
    weighted = cv2.addWeighted(img, 0.55, grid, 0.55, 0.01)
    thresh = 250
    mask = np.where(img < 1.8 * grid, 0.75 * img + 0.25 * grid, 0.25 * img + 0.75 * grid)
    sum = grid + img - 255  # np.where(mask == 0, grid, img * 0.75 + grid * 0.25)
    mask = adjust_gamma(mask.astype(np.uint8), gamma=0.6)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    plt.imshow(mask, cmap='gray')
    plt.show()
    # plt.imshow(grid, cmap='gray')
    # plt.show()
    # plt.imshow(sum, cmap='gray')
    # plt.show()
    # plt.imshow(weighted, cmap='gray')
    # plt.show()


for i in range(25):
    image_number = random.randint(100, 1500)
    grid_number = random.randint(100, 500)
    merge_grid(image_number, grid_number)
