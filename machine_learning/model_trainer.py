from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import config as cfg
import tensorflow as tf
import os
import cv2

print("TensorFlow version:", tf.__version__)

# our training data
liked_path = cfg.liked_images
disliked_path = cfg.disliked_images

def preprocess_images_v2(folder_path: str) -> (list, list):
    images = []
    labels = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        images.append(image)
        labels.append(1 if folder_path == liked_path else 0)
    return images, labels


@tf.autograph.experimental.do_not_convert
def train_tindermodel():
    # keep 50% of the quality otherwise 640x640
    img_width, img_height = 320, 320

    # model settings
    train_data_dir = liked_path
    validation_data_dir = disliked_path
    nb_train_samples = 1378
    nb_validation_samples = 240
    epochs = 5000
    batch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # 3 hidden Layers, will be replaced with resnet or inception
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('softmax'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # if you want to continue training previous weights
    # model.load_weights('model_weights_v1.h5')

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # assign training and test data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # convert to black n white
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('model_weights_v1.h5')
    model.save('tinder_model_main.h5')


train_tindermodel()
