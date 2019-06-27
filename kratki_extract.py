import os
import cv2
import numpy as np
# import scipy.stats
def extract(h,w, iter):
    gen = os.walk('kratki')

    files = next(gen)[2]
    filenames_sampled = np.random.choice(files, size=iter)

    for ind, filename in enumerate(filenames_sampled):
            img = cv2.imread('kratki/'+filename, 0)
            iH, iW = np.shape(img)

            if iH > h and iW > w:
                #start_h = np.random.randint(0, iH - h)
                start_h = np.random.poisson(6, 1)[0]
                if start_h < 0:
                    start_h =0
                if start_h > 11:
                    start_h = 11
                print(start_h)
                start_w=np.random.randint(0, iW-w)
                out = img[start_h:start_h+h, start_w:start_w+w]
                cv2.imwrite('kratki_extracted/'+str(ind)+'.png', out)

extract(48,32,10000)