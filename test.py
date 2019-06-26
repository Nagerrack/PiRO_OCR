from DataAug import ImageDataGenerator
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_func
import keras
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from model import get_model
aug = {
        "rotation_range": 5.0,
        "width_shift_range": 0.05,
        "height_shift_range": 0.05,
        "zoom_range": 0.05,
        "shear_range": 0.05,
        "rescale": 1. / 255,
        "fill_mode": 'nearest'
    }

gen = ImageDataGenerator(preprocessing_function=preprocess_func, rotation_range=aug['rotation_range'], width_shift_range=aug['width_shift_range'],
                         height_shift_range=aug['height_shift_range'], zoom_range=aug['zoom_range'], shear_range=aug['shear_range'],
                         rescale=aug['rescale'], fill_mode=aug['fill_mode'])
g = gen.flow_from_directory('numbers', target_size=(48,32), color_mode='grayscale', classes=[str(i) for i in range(11)]) #[str(i) for i in range(11)])



#model = get_model()


import cv2
while cv2.waitKey(500) != ord('q'):

    cv2.imshow('win', g.next()[0][0])

model = keras.Sequential()
model.add(InputLayer(input_shape=(48,32, 1)))
model.add(Flatten())
model.add(Dense(11))
model.compile(Adam(), loss=binary_crossentropy)
model.fit_generator(g, steps_per_epoch=50)
