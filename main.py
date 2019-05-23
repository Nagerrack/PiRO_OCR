import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    img = cv2.imread('data/img_24.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = remove_background(gray)
    rep = detect_lines(img, thresh)
    rep = broaden_lines(rep)
    result = apply_mask(rep, gray)


    plt.imshow(result, cmap='gray')
    cv2.imwrite('result.png', rep)
    plt.show()

if __name__ == "__main__":
    main()