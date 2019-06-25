import main
import model as md
import nn_data_parser
import numpy as np
import matplotlib.pyplot as plt
import cv2
from preprocessing import preprocess_func
model = md.get_model()

weight_path = 'weights/weights'

model.load_weights(weight_path)


invGamma = 1.0 / 0.25

table = np.array([((i / 255.0) ** invGamma) * 255
                  for i in np.arange(0, 256)]).astype("uint8")

grid_path = "kratki_extracted/"
grids = [cv2.imread(grid_path+'{}.png'.format(i), 0)  for i in range(1500)]

img = cv2.imread('numbers/2/49.png', 0)

rand = np.random.randint(0,1499)
img = preprocess_func(img,grids[rand], table, 0)
img = np.expand_dims(img, axis=-1)
img=img[2:50, 2:34, :]
img = (img-np.mean(img))/np.std(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
plt.imshow(img[0].reshape(48,32), cmap='gray')
plt.show()
print(pred)


