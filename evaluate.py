import model as md
from DataAug import ImageDataGenerator
from preprocessing import preprocess_func

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

weight_path = 'weights/'

model = md.get_model()
model.load_weights(weight_path + 'weights6')
score = model.evaluate_generator(g_eval, steps=100)
print(score)

model2 = md.get_model()
model2.load_weights(weight_path + 'weightsAvg6')
score = model2.evaluate_generator(g_eval, steps=100)
print(score)
