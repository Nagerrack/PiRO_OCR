import numpy as np
import cv2
import matplotlib.pyplot as plt

k1_size = (5, 5)
ta_1 = 9
tuv_1 = 20
k2_size = (9, 9)
ta_2 = 19
tuv_2 = 3

kernel = np.ones((11, 11), np.uint8)

for x in range(1, 30):
    string = "data/img_" + str(x) + ".jpg"
    img = cv2.imread(string, 0)  # read as gray

    # tresh image

    blur_1 = cv2.blur(img, k1_size)
    th = cv2.adaptiveThreshold(blur_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, ta_1, tuv_1)

    # remove grid and threshold artefact's
    blur_2 = cv2.blur(th, k2_size)  #
    th2 = cv2.adaptiveThreshold(blur_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, ta_2, tuv_2)

    # experimental - join text in one shape
    er = cv2.erode(th2, kernel, iterations=30)  # join text
    dil = cv2.dilate(er, kernel, iterations=60)  # destroy point out of text area
    er2 = cv2.erode(dil, kernel, iterations=30)  # rebuild text shape

    # TODO aproximate to rectangle

    # TODO calculate slope factor

    # TODO rotate image

    # Save output
    cv2.imwrite('output/' + str(x) + '.png', er2)
