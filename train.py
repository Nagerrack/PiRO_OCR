import keras
import model

epochs = 100

model = model.get_model()

for i in range(epochs):
    print('epoch {}'.format(i + 1))
    model.fit_generator()
