import keras
from keras.layers import *
from keras import Model
from keras.optimizers import *
from keras.losses import *
from keras.regularizers import *
regL1 = 5e-7
regL2 = 1e-7
def get_model(input_shape=(48, 32, 1), output_category_number=11):
    inputs = Input(input_shape)
    #drop = Dropout(0.25)(inputs)

    conv1_preAct = Conv2D(32, 3, activation='linear', padding='same', kernel_initializer='he_normal', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(inputs)
    bn1 = BatchNormalization()(conv1_preAct)
    conv1 = Activation('relu')(bn1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv1Alternative1_preAct = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(conv1)
    bn2 = BatchNormalization()(conv1Alternative1_preAct)
    conv1Alternative1 = Activation('relu')(bn2)


    conv1Alternative2_preAct = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(conv1Alternative1)
    bn3 = BatchNormalization()(conv1Alternative2_preAct)
    conv1Alternative2 = Activation('relu')(bn3)


    pool1Alternative = MaxPooling2D(pool_size=(2, 2))(conv1Alternative2)

    pool1BranchesAdd = Add()([pool1, pool1Alternative])

    conv2_preAct = Conv2D(64, 3, padding='same', kernel_initializer='he_normal',activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(pool1BranchesAdd)
    bn4 = BatchNormalization()(conv2_preAct)
    conv2 = Activation('relu')(bn4)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv2Alternative1_preAct = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(conv2)
    bn5 = BatchNormalization()(conv2Alternative1_preAct)
    conv2Alternative1 = Activation('relu')(bn5)


    conv2Alternative2_preAct = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(
        conv2Alternative1)
    bn6 = BatchNormalization()(conv2Alternative2_preAct)
    conv2Alternative2 = Activation('relu')(bn6)

    pool2Alternative = MaxPooling2D(pool_size=(2, 2))(conv2Alternative2)

    pool2BranchesAdd = Add()([pool2, pool2Alternative])

    flat = Flatten()(pool2BranchesAdd)

    dense1_preAct = Dense(64, kernel_initializer='he_normal', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(flat)
    bn7 = BatchNormalization()(dense1_preAct)
    dense1 = Activation('relu')(bn7)

    dense2_preAct = Dense(32, kernel_initializer='he_normal', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(dense1)
    bn8 = BatchNormalization()(dense2_preAct)
    dense2 = Activation('relu')(bn8)

    out = Dense(output_category_number, activation='softmax', activity_regularizer=L1L2(regL1, regL2), bias_regularizer=L1L2(regL1, regL2))(dense2)

    model = Model(input=inputs, output=out)
    model.compile(optimizer=Adam(lr=5e-04), loss=categorical_crossentropy, metrics=['accuracy'])

    return model
