from DataAug import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import preprocess_func
import keras
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
gen = ImageDataGenerator(preprocessing_function=preprocess_func)
g = gen.flow_from_directory('numbers', target_size=(48,32), color_mode='grayscale')
plt.imshow(g.next()[0][0].astype(np.uint8))
plt.show()
model = keras.Sequential()
model.add(InputLayer(input_shape=(48,32)))
model.add(Flatten())
model.add(Dense(2))
model.compile(Adam(), loss=binary_crossentropy)
model.fit_generator(g, steps_per_epoch=50)
