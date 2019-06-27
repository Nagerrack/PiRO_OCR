import numpy as np
import cv2

def index_img(image, img_name, path='data/SelfIndexed'):
    img_dictionary = '/NaN/'
    cv2.imshow("img", image)
    k = cv2.waitKey(0)
    if k == ord('1'):
        img_dictionary = '/1/'
    elif k == ord('2'):
        img_dictionary = '/2/'
    elif k == ord('3'):
        img_dictionary = '/3/'
    elif k == ord('4'):
        img_dictionary = '/4/'
    elif k == ord('5'):
        img_dictionary = '/5/'
    elif k == ord('6'):
        img_dictionary = '/6/'
    elif k == ord('7'):
        img_dictionary = '/7/'
    elif k == ord('8'):
        img_dictionary = '/8/'
    elif k == ord('9'):
        img_dictionary = '/9/'
    elif k == ord('0'):
        img_dictionary = '/0/'
    else:
        print('NaN')

    full_path = path + img_dictionary + img_name + '.png'
    print(full_path)
    cv2.imwrite(full_path, img)


if __name__ == "__main__":
    counter = 0
    while 1:
        img = np.zeros((400, 400))
        index_img(img, str(counter))
        counter = counter + 1
