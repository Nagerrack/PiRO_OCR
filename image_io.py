import skimage.io as io
import os


def read_image(path, filename):
    try:
        return io.imread(os.path.join(path, filename))
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