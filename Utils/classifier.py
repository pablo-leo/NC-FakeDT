import keras
from keras.layers import *
from keras import backend as K
from keras.models import Model

def create_Trumpifier(in_size = (1, 70, 100), drop = 0.5,
    loss = 'binary_crossentropy', optimizer = 'SGD',
                     out_size = 2):
    x_in = Input(shape = in_size)

    # Convolutional block
    x = Conv1D(256, 3, activation = 'relu',  kernel_initializer='he_normal')(x_in)
    x = Conv1D(128, 3, activation = 'relu', padding='valid',  kernel_initializer='he_normal')(x)
    x = Conv1D(256, 5, activation = 'relu', padding='same',  kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, 2)(x)
    x = Dropout(drop)(x)
    
    x = Conv1D(128, 3, activation = 'relu',  kernel_initializer='he_normal')(x)
    x = Conv1D(64, 3, activation = 'relu', padding='valid',  kernel_initializer='he_normal')(x)
    x = Conv1D(128, 5, activation = 'relu', padding='same',  kernel_initializer='he_normal')(x)
    x = MaxPooling1D(2, 2)(x)
    x = Dropout(drop)(x)
    
    x = Conv1D(64, 3, activation = 'relu',  kernel_initializer='he_normal')(x_in)
    x = Conv1D(32, 3, activation = 'relu', padding='valid',  kernel_initializer='he_normal')(x)
    x = Conv1D(64, 5, activation = 'relu', padding='same',  kernel_initializer='he_normal')(x)
    x = MaxPooling1D(2, 2)(x)
    x = Dropout(drop)(x)
    
    x = Conv1D(32, 1, padding='same')(x)
    x = Conv1D(16, 1, padding='same')(x)
    x = MaxPooling1D(2, 2)(x)

    x = Flatten()(x)
    x = Dropout(drop)(x)
    
    x = Dense(1120)(x)
    x = Dense(560)(x)
    x = Dense(1120)(x)    
    
    x = Dense(30)(x)
    x = Dense(out_size)(x)
    x_out = Activation('softmax')(x)

    model = Model(x_in, x_out)
    model.compile(optimizer, loss, metrics = ['accuracy'])
    print(model.summary())

    return model

