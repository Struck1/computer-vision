import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Load data


def y2indicator(Y):

    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I


data = pd.read_csv(
    r'C:\Users\ASUS\Desktop\data\Fashion_MNIST\fashion-mnist_train.csv')

data = data.values

np.random.shuffle(data)

X = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
Y = data[:, 0].astype(np.int32)

K = len(set(Y))
Y = y2indicator(Y)
# keras => to_categorical

# make cnn

model = Sequential()

model.add(Conv2D(input_shape = (28, 28, 1), filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())


model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(X, Y, validation_split=0.33, batch_size=32, epochs=2)
print(r)


print(r.history.keys())

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()



