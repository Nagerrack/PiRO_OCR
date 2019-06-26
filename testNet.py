import main
import model as md
import nn_data_parser
import numpy as np
import matplotlib.pyplot as plt
import cv2
from DataAug import ImageDataGenerator
from preprocessing import preprocess_func
model = md.get_model()

weight_path = 'weights/weightsAvg4'

model.load_weights(weight_path)

aug = {
    "rotation_range": 7.5,
    "width_shift_range": 0.13,
    "height_shift_range": 0.13,
    "zoom_range": 0.2,
    "shear_range": 0.05,
    #"rescale": 1. / 255,
    "fill_mode": 'nearest'
}

gen = ImageDataGenerator(preprocessing_function=preprocess_func, rotation_range=aug['rotation_range'],
                         width_shift_range=aug['width_shift_range'],
                         height_shift_range=aug['height_shift_range'], zoom_range=aug['zoom_range'],
                         shear_range=aug['shear_range'],
                          fill_mode=aug['fill_mode'])


g_eval = gen.flow_from_directory('../numbers-test', target_size=(48, 32), classes=[str(i) for i in range(11)],
                                 color_mode='grayscale')

for i in range(1000):
    t = next(g_eval)
    for i in range(len(t[0])):
        toCheck = np.expand_dims(t[0][i], axis=0)
        pred=model.predict(toCheck)
        print(np.argmax(pred))
        plt.imshow(t[0][i][:,:,0])
        plt.show()


#plt.imshow(img[0].reshape(48,32), cmap='gray')
#plt.show()
print(pred)


