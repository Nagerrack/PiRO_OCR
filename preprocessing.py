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


def create_train_test():
    file_path = r"..\numbers — kopia"
    # train_path = r"..\numbers-train"
    test_path = r"..\numbers-test"
    if not path.exists(test_path):
        os.mkdir(test_path)
    for image_dir in os.listdir(file_path):
        for r, _, f in os.walk(path.join(file_path, image_dir)):
            test_files = np.random.choice(f, int(len(f) * 0.12), replace=False)
            if not path.exists(path.join(test_path, image_dir)):
                os.mkdir(path.join(test_path, image_dir))
            for test_file in test_files:
                shutil.move(path.join(file_path, image_dir, test_file), path.join(test_path, image_dir, test_file))


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def prepr2(img):
    img[:,:, 0] = 255 - img[:,:,0]

    return img

import copy as cp
def preprocess_func(img, grid, table=None, clazz=0):
    if clazz != 10:
        # print(clazz)
        #ret, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
        # plt.imshow(img[:,:,0])
        # plt.show()
        img = img.astype(np.uint8)
        c_val = np.random.normal(16, 2.5)
        img2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, c_val)
        # plt.imshow(img2[:, :])
        # plt.show()
        #img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)).astype(np.uint8)
        # plt.imshow(img2[:, :])
        # plt.show()

        img = img/255
        grid = grid/255
        img = cv2.resize(img, (32, 48))
        move = 0

        #img = 1 - img
        mean = np.mean(img)
        img = img - mean
        #img = img[:, 6 + move:42 + move]
        # sigma = np.random.rand()
        # img = cv2.GaussianBlur(img, (3, 3), 1 + sigma)

        #img = adjust_gamma(img, gamma=0.4)
        # plt.imshow(img, cmap='gray')
        # plt.show()

        #img = cv2.GaussianBlur(img, (3, 3), 2)
        #img = adjust_gamma(img, gamma=0.4)

        # plt.imshow(img, cmap='gray')
        # plt.show()


        #grid = adjust_gamma(grid, gamma=0.2)

        #mask = np.where(img < 1.8 * grid, 0.75 * img + 0.25 * grid, 0.25 * img + 0.75 * grid)
        #mask = adjust_gamma(mask.astype(np.uint8), gamma=0.6)
        #imgCp = cp.deepcopy(img)
        mult = np.random.normal(0.65, 0.04)
        imgVal = np.where(img2<32, np.random.poisson(np.mean(grid)*255*mult-16), np.mean(grid*255))/255
        mult2 = np.random.normal(2, 0.5)
        imgVal = cv2.GaussianBlur(imgVal, (5,5), mult2)


        imgCp = img2+mean
        imgCp = np.sqrt(imgCp)
        mask = np.where(img2 < 32, imgVal, grid)
        # plt.imshow(mask[:, :])
        # plt.show()
        # img = cv2.LUT(img, table)
        #
        #
        # div = np.random.randint(110, 135)
        # mean = np.mean(grid) / div
        #
        # thresh = np.random.randint(120, 136)
        # alpha = np.random.randint(89, 95) / 100
        # beta = np.random.randint(90, 100) / 100
        # mask = np.where(img < thresh, (alpha * img + (1 - alpha) * grid) * mean, grid * beta)
        # mask = mask.astype(np.uint8)
        mask = np.expand_dims(mask, axis=-1)
        return mask
    else:
        # img = cv2.resize(img, (52, 52))
        # move = np.random.randint(-1, 1)
        # img = img[:, 6 + move:42 + move]

        return img/255



# create_train_test()
#parse_number_dataset()