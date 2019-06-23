import cv2
import numpy as np


def preprocess(img):
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 21, 10)
    # cv2.imshow("th", th)

    opening = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("morph", opening)

    bitw = cv2.bitwise_not(opening)
    # cv2.imshow("bitw", bitw)

    blur = cv2.medianBlur(bitw,5)
    dilate_blur = cv2.morphologyEx(blur,cv2.MORPH_DILATE,kernel,iterations=1)
    return dilate_blur

def extract(img):
    results = []
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))
    meanArea = np.mean(areas)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > meanArea/2:
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(processedImg, (x, y), (x + w, y + h), (255, 255, 255), 2)
            crop_img = img[y:y + h, x:x + w]
            results.append(crop_img)
    return results


def extract_digits(img):
    processImg = preprocess(img)
    return extract(processImg)

# LOCAL METHODS TO TEST
def test_method():
    for x in range(9, 12):
        string = "data/trainIndex/" + str(x) + ".png"
        img = cv2.imread(string, 0)  # read as gray string = "data/img_" + str(x) + ".3pg3
        result = extract_digits(img)
        for item in result:
            cv2.imshow("img",item)
            wait_for_key()

def wait_for_key():
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_method()