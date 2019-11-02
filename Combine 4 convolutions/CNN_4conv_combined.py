#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:06:06 2019

@author: antoinek
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# Import created function
from Test_concatenate import concatenate_4to1


# Import and Reshape the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# input image dimensions
img_rows, img_cols = 28, 28
batch_size =128;

# Change to 1 hot encoder
y_train = to_categorical(y_train)
#y_test_integer = y_test
y_test = to_categorical(y_test)

x_train = x_train.reshape (60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)
input_shape = (img_rows, img_cols, 1)

# This returns a tensor
inputs = Input(shape=input_shape)

#4 conv to merge back
conv_A= Conv2D(1, kernel_size=(2,2), activation='relu')(inputs)
conv_B= Conv2D(1, kernel_size=(2,2), activation='relu')(inputs)
conv_C= Conv2D(1, kernel_size=(2,2), activation='relu')(inputs)
conv_D= Conv2D(1, kernel_size=(2,2), activation='relu')(inputs)

#Input shape-1, because the output of conv will be 1 dimenssion smaller
concatenated_4conv =concatenate_4to1(conv_A,conv_B,conv_C,conv_D,27)
#concatenated_pool = MaxPooling2D(pool_size=(2,2))(concatenated_4conv)

conv1 = Conv2D(32, kernel_size=(3,3), activation='relu')(concatenated_4conv)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
hidden1 = Dense(128, activation='relu')(flat)
output = Dense(10, activation='softmax')(hidden1)
model = Model(inputs=inputs, outputs=output)

# Save model
#model.save("model_nopooling.h5")
### Load previous model
model = load_model('model_pooling.h5')

# summarize layers
print(model.summary())
plot_model(model,show_shapes= True, to_file='cnn.png')

opti= Adam(learning_rate=0.01)
model.compile(optimizer=opti,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs= 4, batch_size= batch_size, verbose=1, validation_data=(x_test, y_test))  






