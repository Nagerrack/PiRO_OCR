import os
import cv2
import numpy as np
def extract(h,w, iter):
    gen = os.walk('kratki')

    files = next(gen)[2]
    filenames_sampled = np.random.choice(files, size=iter)

    for ind, filename in enumerate(filenames_sampled):
            img = cv2.imread('kratki/'+filename, 0)
            iH, iW = np.shape(img)

            if iH > h and iW > w:
                start_h = np.random.randint(0, iH - h)
                start_w=np.random.randint(0, iW-w)
                out = img[start_h:start_h+h, start_w:start_w+w]
                cv2.imwrite('kratki_extracted/'+str(ind)+'.jpg', out)

extract(48,32,1000)