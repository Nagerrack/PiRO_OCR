import keras
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

gen = ImageDataGenerator(preprocessing_function=preprocess_func, rotation_range=aug['rotation_range'], width_shift_range=aug['width_shift_range'],
                         height_shift_range=aug['height_shift_range'], zoom_range=aug['zoom_range'], shear_range=aug['shear_range'],
                         rescale=aug['rescale'], fill_mode=aug['fill_mode'])
g = gen.flow_from_directory('numbers', target_size=(48,32), classes=[str(i) for i in range(11)], color_mode='grayscale')

epochs = 100

model = md.get_model()

avg = model.get_weights()
alpha = 0.95
print(model.metrics_names)
for i in range(epochs):
    print('epoch {}'.format(i + 1))
    model.fit_generator(g, steps_per_epoch=50, epochs=1)
    current = model.get_weights()
    for i in range(len(avg)):
        avg[i] = avg[i] * alpha + (1-alpha) * current[i]

model.save_weights('weights')
score = model.evaluate_generator(g, steps=50)
print(score)

model2 = md.get_model()
model2.set_weights(avg)
model2.save_weights('weightsAvg')
score = model2.evaluate_generator(g, steps=50)
print(score)






