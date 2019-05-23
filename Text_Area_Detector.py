import numpy as np
import cv2
import matplotlib.pyplot as plt



def show(img, img2 =None):
    im = cv2.resize(img, (0,0), fx=0.3, fy =0.3)
    if img2 is not None:
        im2 = cv2.resize(img2, (0,0), fx=0.3, fy =0.3)
    while cv2.waitKey(30) != ord('q'):
        cv2.imshow('win', im)
        if img2 is not None:
            cv2.imshow('win2', im2)

def detect():
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
        # show(th)
        # remove grid and threshold artefact's
        blur_2 = cv2.blur(th, k2_size)  #
        th2 = cv2.adaptiveThreshold(blur_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, ta_2, tuv_2)
        # show(th2)
        # experimental - join text in one shape
        er = cv2.erode(th2, kernel, iterations=30)  # join text
        # show(er)
        dil = cv2.dilate(er, kernel, iterations=60)  # destroy point out of text area
        # show(dil)
        er2 = cv2.erode(dil, kernel, iterations=30)  # rebuild text shape
        # show(er2, img)

        res = np.where(er2 < 127, img, 0)
        # show(res)
        # TODO aproximate to rectangle
        mask = np.where(er2 < 127)
        max1, max2= np.max(mask, axis=-1)
        min1, min2 =np.min(mask, axis=-1)
        min1 = max(int(min1 - img.shape[0] * 0.03), 0)
        min2 = max(int(min2 - img.shape[1] * 0.03), 0)
        max1 = min(int(max1 + img.shape[0] * 0.03), img.shape[0] - 1)
        max2 = min(int(max2 + img.shape[1] * 0.03), img.shape[1] - 1)
        rect = img[min1:max1, min2:max2]
        # TODO calculate slope factor

        # TODO rotate image

        # Save output
        cv2.imwrite('output/' + str(x) + '.png', er2)

def detect_one_img(img):
    k1_size = (5, 5)
    ta_1 = 9
    tuv_1 = 20
    k2_size = (9, 9)
    ta_2 = 19
    tuv_2 = 3
    kernel = np.ones((11, 11), np.uint8)

    #img = np.uint8(img/np.max(img)*255)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_1 = cv2.blur(img, k1_size)
    th = cv2.adaptiveThreshold(blur_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, ta_1, tuv_1)
    # show(th)
    # remove grid and threshold artefact's
    blur_2 = cv2.blur(th, k2_size)  #
    th2 = cv2.adaptiveThreshold(blur_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, ta_2, tuv_2)
    # show(th2)
    # experimental - join text in one shape
    er = cv2.erode(th2, kernel, iterations=30)  # join text
    # show(er)
    dil = cv2.dilate(er, kernel, iterations=60)  # destroy point out of text area
    # show(dil)
    er2 = cv2.erode(dil, kernel, iterations=30)  # rebuild text shape
    # show(er2, img)

    mask = np.where(er2 < 127)
    max1, max2 = np.max(mask, axis=-1)
    min1, min2 = np.min(mask, axis=-1)

    min1 = max( int(min1 - img.shape[0] * 0.03), 0)
    min2 = max(int(min2 - img.shape[1] * 0.06), 0)
    max1 = min(int(max1 + img.shape[0] * 0.03), img.shape[0]-1)
    max2 = min(int(max2 + img.shape[1] * 0.06), img.shape[1]-1)
    rect = img[min1:max1, min2:max2]
    # show(rect)
    # res = np.where(er2 < 127, img, 0)
    # show(res)
    return rect
if __name__ == "__main__":
    detect()