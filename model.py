import keras
from keras.layers import *


def get_model(input_shape=(32, 48, 1), output_category_number=10):
    model = keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))

    # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    # fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    # one more dropout for convergence' sake :)
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(output_category_number, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
