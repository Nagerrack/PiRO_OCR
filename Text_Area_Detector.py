import cv2
import numpy as np


def show(img, img2=None):
    im = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    if img2 is not None:
        im2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3)
    while cv2.waitKey(30) != ord('q'):
        cv2.imshow('win', im)
        if img2 is not None:
            cv2.imshow('win2', im2)


def naive_approximate_rectanlge_points(img, margin, lmargin=0.04):
    mask = np.where(img < 127)
    max1, max2 = np.max(mask, axis=-1)
    min1, min2 = np.min(mask, axis=-1)
    min1 = max(int(min1 - img.shape[0] * margin), 0)
    min2 = max(int(min2 - img.shape[1] * lmargin), 0)
    max1 = min(int(max1 + img.shape[0] * margin), img.shape[0] - 1)
    max2 = min(int(max2 + img.shape[1] * margin), img.shape[1] - 1)
    return min1, max1, min2, max2


def detect():
    k1_size = (5, 5)
    ta_1 = 9
    tuv_1 = 20
    k2_size = (9, 9)
    ta_2 = 19
    tuv_2 = 5
    er_iter_1 = 30
    er_iter_2 = 30
    dil_iter_1 = 60
    r_margin = 0.1
    kernel = np.ones((11, 11), np.uint8)

    for x in range(1, 30):
        string = "data/img_" + str(x) + ".jpg"
        img = cv2.imread(string, 0)  # read as gray string = "data/img_" + str(x) + ".jpg"
        # color = cv2.imread(string, 3)  # read as gray
        # color = remove_surrounding(color)
        # img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # tresh image
        blur_1 = cv2.blur(img, k1_size)
        th = cv2.adaptiveThreshold(blur_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, ta_1, tuv_1)

        # remove grid and threshold artefact's
        blur_2 = cv2.blur(th, k2_size)  #
        th2 = cv2.adaptiveThreshold(blur_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, ta_2, tuv_2)
        # ret,th2 = cv2.threshold(blur_2,190,255,cv2.THRESH_BINARY)

        # experimental - join text in one shape
        er = cv2.erode(th2, kernel, iterations=er_iter_1)  # join text
        dil = cv2.dilate(er, kernel, iterations=dil_iter_1)  # destroy point out of text area
        er2 = cv2.erode(dil, kernel, iterations=er_iter_2)  # rebuild text

        # create rectangle mask to remove outliers
        min1, max1, min2, max2 = naive_approximate_rectanlge_points(er2, r_margin)
        mask = np.full(img.shape, 255, np.uint8)
        mask[min1:max1, min2:max2] = 0

        # apply mask
        result = cv2.bitwise_or(th2, mask)
        result = cv2.bitwise_not(result)

        # approximate to rectangle above rest text
        pts = cv2.findNonZero(result)
        ret = cv2.minAreaRect(pts)

        (cx, cy), (w, h), ang = ret
        # print(cx)
        # print(cy)
        # print(w)
        # print(h)
        if w > h:
            w, h = h, w
            ang += 90
        # Draw text box
        box = cv2.boxPoints(ret)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = box.astype(int)
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        print(box)
        print(box[1, 1])
        # rotate image
        # M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        # rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # print(box[2])
        # print(rotated_fragment)
        print("------------")
        # Save output
        # cv2.imwrite('output/rot_' + str(x) + '.png', rotated_fragment)
        cv2.imwrite('output/res_' + str(x) + '.png', result)
        # cv2.imwrite('output/' + str(x) + '.png', rotated)


def detect_one_img(img, retNaive=False):
    img = img.copy()
    k1_size = (5, 5)
    ta_1 = 9
    tuv_1 = 20
    k2_size = (9, 9)
    ta_2 = 19
    tuv_2 = 3
    er_iter_1 = 30
    er_iter_2 = 30
    dil_iter_1 = 70
    r_margin = 0.1
    kernel = np.ones((11, 11), np.uint8)

    # img = np.uint8(img/np.max(img)*255)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_1 = cv2.blur(img, k1_size)
    th = cv2.adaptiveThreshold(blur_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, ta_1, tuv_1)
    # remove grid and threshold artefact's
    blur_2 = cv2.blur(th, k2_size)  #
    th2 = cv2.adaptiveThreshold(blur_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, ta_2, tuv_2)
    # experimental - join text in one shape
    er = cv2.erode(th2, kernel, iterations=er_iter_1)  # join text
    dil = cv2.dilate(er, kernel, iterations=dil_iter_1)  # destroy point out of text area
    er2 = cv2.erode(dil, kernel, iterations=er_iter_2)  # rebuild text

    # create rectangle mask to remove outliers
    min1, max1, min2, max2 = naive_approximate_rectanlge_points(er2, r_margin, lmargin=r_margin)
    mask = np.full(img.shape, 255, np.uint8)
    mask[min1:max1, min2:max2] = 0

    # apply mask
    result = cv2.bitwise_or(th2, mask)
    result = cv2.bitwise_not(result)

    # approximate to rectangle above rest text
    pts = cv2.findNonZero(result)
    ret = cv2.minAreaRect(pts)

    (cx, cy), (w, h), ang = ret

    if w > h:
        w, h = h, w
        ang += 90
    # Draw text box
    box = cv2.boxPoints(ret)  # cv2.boxPoints(rect) for OpenCV 3.x

    #M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    if retNaive:
        min1, max1, min2, max2 = naive_approximate_rectanlge_points(er2, r_margin, lmargin=0.04)
        rect = img[min1:max1, min2:max2]
        return rect, box, ang, (min1, max1, min2, max2)
    return box, ang


if __name__ == "__main__":
    detect()
