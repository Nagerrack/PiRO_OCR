import main
import model as md
import nn_data_parser
import numpy as np
import matplotlib.pyplot as plt

from DataAug import ImageDataGenerator
from preprocessing import preprocess_func

model = md.get_model()

weight_path = 'weights/weightsV2-1'

model.load_weights(weight_path)

indice_images = main.get_indices(3)

c = 0

indice_images.pop(0)
for index in indice_images:
    plt.imshow(index, cmap='gray')
    plt.show()

    windows = nn_data_parser.slide_window(index)

    prediction_list = []
    for window in windows:
        new_window = np.array(window, dtype=np.float32)
        new_window /= 255.0

        new_window = np.expand_dims(new_window, axis=-1)
        new_window = np.expand_dims(new_window, axis=0)

        # prediction_list.append(model.predict(new_window))
        prediction = model.predict(new_window)
        #print(prediction)
        prediction_list.append(np.argmax(prediction))
        # plt.imshow(window, cmap='gray')
        # plt.show()
        # print()
        # break

    print(prediction_list)



