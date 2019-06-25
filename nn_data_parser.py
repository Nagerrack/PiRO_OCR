import numpy as np
import cv2
import matplotlib.pyplot as plt


def slide_window(img, step_size: int = 4, w_height: int = 48, w_width: int = 32,create_image_with_part_step=True):
    h = img.shape[0]
    w = img.shape[1]
    assert h > w_height, "Image height is less than window height"
    assert w > w_width, "Image width is less than window width"
    assert step_size > 1, "Step size must be grater than 1"

    itr_number = (w - w_width) // step_size + 1
    missing_full_step_pixels = (w - w_width) % step_size

    vccp = (h - w_height) // 2  # vertical_crop_corner_position
    result = []
    for i in range(itr_number):
        hccp = i * step_size
        crop_img = img[vccp:vccp + w_height, hccp:hccp + w_width]
        result.append(crop_img)

    if create_image_with_part_step and missing_full_step_pixels > 0:
        crop_img = img[vccp:vccp + w_height, w - w_width:w]
        result.append(crop_img)
    return result

# TEST SECTION
if __name__ == "__main__":
    img = cv2.imread("data/raw_index/1.png")
    imgs = slide_window(img, w_height=100, w_width=50)
    plt.imshow(img, cmap='gray')
    plt.show()
    for item in imgs:
        plt.imshow(item, cmap='gray')
        plt.show()
