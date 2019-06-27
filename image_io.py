import os

import cv2
# import matplotlib.pyplot as plt


def read_image(path, filename):
    try:
        return cv2.imread(os.path.join(path, filename))
    except:
        print("Can't read file")


def read_particular_images(path, indices):
    imgs = []

    for i in indices:
        imgs.append(read_image(path, "img_" + str(i) + ".jpg"))

    return imgs


def read_entire_data_set(path, pic_count):
    pics = []

    for i in range(1, pic_count + 1):
        pics.append(read_image(path, "img_" + str(i) + ".jpg"))

    return pics


# def display_image(img, i):
#     plt.imshow(img, cmap='gray')
#
#     cv2.imwrite(str(i) + '.png', img)
#
#     plt.axis("off")
#     plt.show()
