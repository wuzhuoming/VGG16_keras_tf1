from tensorflow import keras
import tensorflow as tf
import os,json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pathlib
import random
from tensorflow.python.data.experimental import AUTOTUNE


BATCH_SIZE = 32

def build_and_compile_cnn_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         'worker': ["localhost:1111", "localhost:1112"]
#     },
#     'task': {'type': 'worker', 'index': 0}
# })
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
#
# trdata = ImageDataGenerator()
# traindata = trdata.flow_from_directory(directory="cats_and_dogs_filtered/train",target_size=(224,224))
# tsdata = ImageDataGenerator()
# testdata = tsdata.flow_from_directory(directory="cats_and_dogs_filtered/validation", target_size=(224,224))
#
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#
# with strategy.scope():
#     model = build_and_compile_cnn_model()
# hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])

train_data_root = pathlib.Path("./cats_and_dogs_filtered/train")
test_data_root = pathlib.Path("./cats_and_dogs_filtered/validation")

train_image_paths = list(train_data_root.glob('*/*'))
train_image_paths = [str(path) for path in train_image_paths]
random.shuffle(train_image_paths)
test_image_paths = list(test_data_root.glob('*/*'))
test_image_paths = [str(path) for path in test_image_paths]
random.shuffle(test_image_paths)
train_image_count = len(train_image_paths)
test_image_count = len(test_image_paths)


label_names = sorted(item.name for item in train_data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]
test_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_image_paths]


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range
  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


train_path_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
test_path_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)
train_image_ds = train_path_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_path_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_image_labels, tf.int64))
test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_image_labels, tf.int64))

train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))

train_ds = train_image_label_ds.shuffle(buffer_size=train_image_count)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = test_image_label_ds.shuffle(buffer_size=test_image_count)
test_ds = test_ds.repeat()
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:1111", "localhost:1112"]
    },
    'task': {'type': 'worker', 'index': 1}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = build_and_compile_cnn_model()
model.fit(train_ds,validation_steps=10,validation_data=test_ds,epochs=10, steps_per_epoch=3)