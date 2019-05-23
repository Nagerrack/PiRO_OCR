import numpy as np
from image_io import  *
from surrounding_removal import remove_surrounding
from Text_Area_Detector import show, detect_one_img

def remove_background(img):
    # filtr medianowy i adaptatywne progowanie
    blur_gray = cv2.medianBlur(img, 7)
    thresh = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 20)

    return thresh


def detect_lines(img, thresh):
    # suma w wierszach i przeskalowanie zakresu wartości
    integral = np.sum(thresh, axis=1)
    integral = integral / np.max(integral) * 255

    # utworzenie obrazu o oryginalnym rozmiarze
    rep = np.transpose(np.tile(integral, (img.shape[1], 1)))
    rep = np.uint8(rep / np.max(rep) * 255)

    return rep


def broaden_lines(rep):
    # progowanie w punkcie średniej obrazu i rozszerzenie obszarów białych
    ret, rep = cv2.threshold(rep, np.mean(rep), 255, cv2.THRESH_BINARY)
    # show(rep)
    rep=cv2.erode(rep, kernel=np.ones(shape=(17,1)))
    # show(rep)
    rep = cv2.dilate(rep, kernel=np.ones(shape=(25, 1)))
    # show(rep)
    rep = cv2.erode(rep, kernel=np.ones(shape=(11, 1)))
    # show(rep)
    rep = cv2.dilate(rep, kernel=np.ones(shape=(3, 1)))
    # show(rep)
    return rep


def apply_mask(rep, img):
    return np.where(rep < 127, img, 0)


def main():
    for i in range(1,30):
        imgs = read_particular_images("data", [i])
        # imgs = read_entire_data_set("data", 29)
        # show(imgs[0])
        img = remove_surrounding(imgs[0])
        # show(img)
        img = detect_one_img(img)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # show(img)
        thresh = remove_background(img)
        # show(thresh)
        rep = detect_lines(img, thresh)
        # show(rep)
        rep = broaden_lines(rep)
        show(rep)

        result = apply_mask(rep, img)
        # show(result)
        #display_image(result, i)


if __name__ == "__main__":
    main()
