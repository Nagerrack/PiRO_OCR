import model as md
from DataAug import ImageDataGenerator
from preprocessing import preprocess_func

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

g_eval = gen.flow_from_directory('../numbers-test', target_size=(48, 32), classes=[str(i) for i in range(11)],
                                 color_mode='grayscale')

weight_path = 'weights/'
name_format = 'weightsV3-8_original_nodrop'

model = md.get_model()
model.load_weights(weight_path + name_format)
score = model.evaluate_generator(g_eval, steps=100)
print(score)

model2 = md.get_model()
model2.load_weights(weight_path + name_format.replace('weights', 'weightsAvg'))
score = model2.evaluate_generator(g_eval, steps=100)
print(score)
