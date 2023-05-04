from keras import backend as K
import config as cfg
import tensorflow as tf
import os
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inceptionresnetv2_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocess_input
from keras.utils import to_categorical


def preprocess_images(folder_path: str) -> (list, list):
    images = []
    labels = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        images.append(image)
        labels.append(1 if folder_path == cfg.liked_images else 0)
    return images, labels


def build_model(model_name: str) -> (Sequential, ImageDataGenerator):
    # models am interested in running base weights are static imagenet  weights
    if model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
        preprocess_input = vgg16_preprocess_input
    elif model_name == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
        preprocess_input = inceptionv3_preprocess_input
    elif model_name == 'xception':
        base_model = Xception(weights='imagenet', include_top=False)
        preprocess_input = xception_preprocess_input
    elif model_name == 'inceptionresnetv2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
        preprocess_input = inceptionresnetv2_preprocess_input
    else:
        raise ValueError('Invalid model name or not implemented!')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    # since we can have multiple classes, we use categorical_crossentropy otherwise if we want liner output we use binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, preprocess_input


def train_tinder_model(liked_path: str, disliked_path: str, model_name: str) -> Sequential:
    # Load and preprocess the 'liked' images
    liked_images, liked_labels = preprocess_images(liked_path)

    # Load and preprocess the 'disliked' images
    disliked_images, disliked_labels = preprocess_images(disliked_path)

    # Concatenate the data and convert the labels to one-hot encoding
    X = np.concatenate([liked_images, disliked_images])
    y = to_categorical(np.concatenate([liked_labels, disliked_labels]))

    # Split the data into training and testing sets (e.g. 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build the model
    model, preprocess_input = build_model(model_name)

    # Preprocess the images using the appropriate function
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    # Train the model on the training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    # save model
    model.save('tinder_model_main.h5')
    # save model weights
    model.save_weights('tinder_model_weights.h5')
    return model


def main():
    # run training
    train_tinder_model(cfg.liked_images,
                       cfg.disliked_images,
                       args.model_name)


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vgg16', help='Model name')
    args = parser.parse_args()

    main()
