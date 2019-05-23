import numpy as np
from image_io import  *


def remove_background(img):
    # filtr medianowy i adaptatywne progowanie
    blur_gray = cv2.medianBlur(img, 5)
    thresh = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 20)

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
    # rep=cv2.dilate(rep, kernel=np.ones(shape=(11,1)))

    return rep


def apply_mask(rep, img):
    return np.where(rep < 127, img, 0)


def main():
    imgs = read_particular_images("data", [24])
    # imgs = read_entire_data_set("data", 29)

    gray = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)

    thresh = remove_background(gray)
    rep = detect_lines(imgs[0], thresh)
    rep = broaden_lines(rep)
    result = apply_mask(rep, gray)

    display_image(result, 24)


if __name__ == "__main__":
    main()