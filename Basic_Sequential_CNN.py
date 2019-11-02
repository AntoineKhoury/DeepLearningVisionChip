# Multiple libraries required to import, visualize and treat the data
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

#already splitted in train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# =============================================================================
# # Show what these images corresponds to
# i=0
# plt.imshow(x_train[i], cmap=plt.cm.gray_r)
# print("\n",y_train[i])
# =============================================================================


## Process the data
X_train = x_train.reshape(60000,28,28,1)
X_test = x_test.reshape(10000,28,28,1)


# Change to 1 hot encoder
y_train = to_categorical(y_train)
y_test_integer = y_test
y_test = to_categorical(y_test)


#create the model
CNN = Sequential()
#add model layers
CNN.add(Conv2D(filters= 16, kernel_size=5, activation='relu', input_shape=(28,28,1)))
CNN.add(Conv2D(filters=8, kernel_size=6, activation='relu'))
CNN.add(Flatten())
CNN.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
CNN.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
y_predicted = CNN.predict(X_test)



import numpy as np  
y_predicted_integer  = np.argmax(y_predicted, axis=1)


# =============================================================================
# # Show what this corresponds to
# i = 10
# plt.imshow(x_test[i], cmap=plt.cm.gray_r)
# print("The algorithm predicted: \n",y_predicted[i])
# print("The true value is: \n",y_test[i])
# =============================================================================

# ###### Get a sense of how well our algorithm did
# =============================================================================
# from sklearn.metrics import confusion_matrix
# 
# error = confusion_matrix(y_test, y_predicted)
# print ("The Confusion matrix is: \n",error)
# =============================================================================


