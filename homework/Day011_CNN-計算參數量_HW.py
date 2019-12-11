from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model

# CNN
classifier = Sequential()
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
print(classifier.summary())

# FC
classifier = Sequential()
inputs = Input(shape=(784,))
x = Dense(units=288)(inputs)
model = Model(inputs=inputs, outputs=x)
print(model.summary())


# CNN
classifier = Sequential()
classifier.add(Convolution2D(filters=64, kernel_size=(5, 5), input_shape=(20, 20, 1)))
print(classifier.summary())

# FC
classifier = Sequential()
inputs = Input(shape=(400,))
x = Dense(units=1664)(inputs)
model = Model(inputs=inputs, outputs=x)
print(model.summary())