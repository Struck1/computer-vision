from tensorflow.keras.layers import Input, Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from builtins import range, input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.models import Sequential
from util import plot_confusion_matrix


# re-size all the images to this

IMAGE_SIZE = [100, 100]

# train config

epochs = 5
batchs_size = 32

# https://www.kaggle.com/moltean/fruits
train_path = '../data_files/fruits-360-small/Training'
test_path = '../data_files/fruits-360-small/Test'

# useful for getting number of files

image_files = glob(train_path + '/*/*.jp*g')
test_image_files = glob(test_path + '/*/*.jp*g')

print(len(image_files))

print(
    f'image_files type : {type(image_files)}, image_files len : {len(image_files)}')
print(image_files[0])
# useful for getting number of classes

folders = glob(train_path + '/*')
print("class :", len(folders))

# look at an image

# plt.imshow(image.img_to_array(image.load_img(
#     np.random.choice(image_files))).astype('uint8'))
# plt.show()


# vgg preprocessing layer
vgg = VGG16(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

model = Sequential()
# our layers

x = Flatten()(vgg.output)

# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object

model = Model(inputs=vgg.input, outputs=prediction)
print(vgg.input)

# view the structure of the model
print(model.summary())

# optimization

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# create an instance of ImageDataGenerator

ımage_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)


test_gen = ımage_generator.flow_from_directory(
    test_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
    labels[v] = k
# batch size = 32
#shuffle = true
# should be a strangely colored image (due to VGG weights being BGR)


for x, y in test_gen:
    print(x[0].shape, y[0])
    print("min:", x[0].min(), "max:", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

# create generators

train_generators = ımage_generator.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,

)

test_generators = ımage_generator.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
)

print(len(image_files) // batchs_size)
# fit model
print(train_generators.class_indices)
print(train_generators.classes.shape)
r = model.fit_generator(
    train_generators,
    validation_data=test_generators,
    epochs=1,
    steps_per_epoch=len(image_files) // batchs_size,
    validation_steps=len(test_image_files) // batchs_size,
)


def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in ımage_generator.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batchs_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(test_path, len(test_image_files))
print(valid_cm)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')



