# import keras
import model as md
from DataAug import ImageDataGenerator
from preprocessing import preprocess_func, prepr2
# import tensorflow as tf

import numpy as np

aug = {
    "rotation_range": 7.5,
    "width_shift_range": 0.13,
    "height_shift_range": 0.13,
    "zoom_range": 0.2,
    "shear_range": 0.05,
    # "rescale": 1. / 255,
    "fill_mode": 'nearest'
}

gen = ImageDataGenerator(preprocessing_function=preprocess_func, rotation_range=aug['rotation_range'],
                         width_shift_range=aug['width_shift_range'],
                         height_shift_range=aug['height_shift_range'], zoom_range=aug['zoom_range'],
                         shear_range=aug['shear_range'],
                         fill_mode=aug['fill_mode'])
g = gen.flow_from_directory('../numbers-train', target_size=(48, 32), classes=[str(i) for i in range(11)],
                            color_mode='grayscale', batch_size=64)

# g=gen.flow(x_train, y_train, batch_size=128)
# import cv2
# while cv2.waitKey(500) != ord('q'):
#
#     cv2.imshow('win', g.next()[0][0])


weight_path = 'weights/'
name_format = 'weightsV3-6_original_nodrop'

epochs = 60

model = md.get_model()

avg = model.get_weights()
alpha = 0.825
print(model.metrics_names)
for i in range(epochs):
    print('epoch {}'.format(i + 1))
    model.fit_generator(g, epochs=1)
    current = model.get_weights()
    for j in range(len(avg)):
        avg[j] = avg[j] * alpha + (1 - alpha) * current[j]

model.save_weights(weight_path + name_format)

g_eval = gen.flow_from_directory('../numbers-test', target_size=(48, 32), classes=[str(i) for i in range(11)],
                                 color_mode='grayscale')



score = model.evaluate_generator(g_eval)
print(score)

model2 = md.get_model()
model2.set_weights(avg)
model2.save_weights(weight_path + name_format.replace('weights', 'weightsAvg'))
score = model2.evaluate_generator(g_eval)
print(score)
