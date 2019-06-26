import main
import model as md
import nn_data_parser
import numpy as np
import matplotlib.pyplot as plt

from DataAug import ImageDataGenerator
from preprocessing import preprocess_func

model = md.get_model()

weight_path = 'weights/weights'

model.load_weights(weight_path)

indice_images = main.get_indices(3)

# c = 0
#
# indice_images.pop(0)
# for index in indice_images:
#     plt.imshow(index, cmap='gray')
#     plt.show()
#
#     windows = nn_data_parser.slide_window(index)
#
#     prediction_list = []
#     for window in windows:
#         new_window = np.array(window, dtype=np.float32)
#         new_window /= 255.0
#         plt.imshow(new_window, cmap='gray')
#         plt.show()
#         new_window = np.expand_dims(new_window, axis=-1)
#         new_window = np.expand_dims(new_window, axis=0)
#
#         # prediction_list.append(model.predict(new_window))
#         prediction = model.predict(new_window)
#         prediction_list.append(np.argmax(prediction))
#
#         # print()
#         # break
#
#     print(prediction_list)


aug = {
    "rotation_range": 5.0,
    "width_shift_range": 0.05,
    "height_shift_range": 0.05,
    "zoom_range": 0.05,
    "shear_range": 0.05,
    "rescale": 1. / 255,
    "fill_mode": 'constant'
}

gen = ImageDataGenerator(preprocessing_function=preprocess_func, rotation_range=aug['rotation_range'],
                         width_shift_range=aug['width_shift_range'],
                         height_shift_range=aug['height_shift_range'], zoom_range=aug['zoom_range'],
                         shear_range=aug['shear_range'],
                         rescale=aug['rescale'], fill_mode=aug['fill_mode'])
g_eval = gen.flow_from_directory('../numbers-test', target_size=(48, 32), classes=[str(i) for i in range(11)],
                                 color_mode='grayscale')

score = model.evaluate_generator(g_eval, steps=50)
print(score)
