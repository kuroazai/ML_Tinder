import config as cfg
import tensorflow as tf
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inceptionresnetv2_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_model(model_name: str, input_shape: tuple) -> (Sequential, ImageDataGenerator):
    # models am interested in running base weights are static imagenet  weights
    if model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = vgg16_preprocess_input
    # elif model_name == 'resnet50':
    #     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    #     preprocess_input = resnet50_preprocess_input
    elif model_name == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = inceptionv3_preprocess_input
    elif model_name == 'xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = xception_preprocess_input
    elif model_name == 'inceptionresnetv2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = inceptionresnetv2_preprocess_input
    else:
        raise ValueError('Invalid model name')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    # TODO: predict the number of classes which should be passed in
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    # since we can have multiple classes, we use categorical_crossentropy otherwise if we want liner output we use binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, preprocess_input


def train_tinder_model(staging_path: str, model_name: str) -> Sequential:
    # Create an image generator to load and preprocess the images on the fly
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    # flow from staging area to get training data
    train_generator = datagen.flow_from_directory(
        staging_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    # flow from stagin area to get test data
    test_generator = datagen.flow_from_directory(
        staging_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    # Build the model and pass the shape
    model, preprocess_input = build_model(model_name, input_shape=(224, 224, 3))
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Callbacks for data visualization in TensorBoard
    callbacks = [
        ModelCheckpoint('tinder_model.h5', save_best_only=True),
        EarlyStopping(patience=3),
        ReduceLROnPlateau(factor=0.1, patience=2)
    ]

    # Train the model on the training data and test on test data
    model.fit(train_generator,
              epochs=10,
              validation_data=test_generator,
              callbacks=callbacks)
    # save model
    model.save('tinder_model_main.h5')
    # save model weights
    model.save_weights('tinder_model_weights.h5')
    # not sure why but avaliable if needed
    return model


def main():
    # run training
    train_tinder_model(cfg.staging_area,
                       args.model_name)


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vgg16', help='Model name')
    args = parser.parse_args()

    main()
