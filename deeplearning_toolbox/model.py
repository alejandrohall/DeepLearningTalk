import numpy as np

from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from keras.utils import multi_gpu_model
from collections import Counter


class DeepLearningModel:

    def __init__(self, model, dataset_path):
        self.model = model
        self.dataset_path = dataset_path
        self.input_shape = None

    def finetune(self, input_shape, full_train=False, layers_to_finetune=1):
        self.input_shape = input_shape

        output_shape = len(list((Path(self.dataset_path) / 'train').iterdir()))

        popped_layer = self.model.layers.pop()
        outputs = Dense(output_shape, activation=popped_layer.activation)(self.model.output)

        self.model = Model(inputs=self.model.inputs, outputs=outputs)

        if not full_train:
            for layer in self.model.layers:
                layer.trainable = False

            trainable_layers = [layer for layer in self.model.layers if layer.count_params() > 0]
            for layer in trainable_layers[-layers_to_finetune:]:
                layer.trainable = True

    def paralelize(self, gpu_numbers):
        self.model = multi_gpu_model(self.model, gpus=gpu_numbers)

    def get_layers_params(self):
        layers_name = [layer.name for layer in self.model.layers]
        layers_params = [layer.count_params() for layer in self.model.layers]

        return list(zip(layers_name, layers_params))

    def compile(self, optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, epochs=1, batch_size=64, imbalanced=False, early_stopping=False):
        callbacks = None
        class_weights = None

        train_generator, valid_generator = self.get_data_generator(self.input_shape, batch_size)

        if early_stopping:
            callbacks = [EarlyStopping(patience=2)]

        if imbalanced:
            counter = Counter(train_generator.classes)
            max_val = float(max(counter.values()))
            class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
            print(class_weights)

        return self.model.fit_generator(train_generator,
                                        epochs=epochs,
                                        validation_data=valid_generator,
                                        callbacks=callbacks,
                                        class_weight=class_weights)

    def predict(self, x):
        return self.model.predict(x)

    def predict_generator(self, generator, class_names=False):
        index_to_class = {v: k for k, v in generator.class_indices.items()}

        probabilities = self.model.predict_generator(generator)
        predictions = probabilities.argmax(axis=-1)
        real = generator.classes

        if class_names:
            predictions = [index_to_class[index] for index in predictions]
            real = [index_to_class[index] for index in real]

        return real, predictions, probabilities

    def save_model(self, name='skin.h5'):
        self.model.save(name)

    def save_weights(self, name='model_weights.h5'):
        self.model.save_weights(name)

    def load_weights(self, name_weights):
        self.model.load_weights(name_weights)

    def get_data_generator(self, input_shape, batch_size):

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            vertical_flip=True,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            (Path(self.dataset_path) / 'train').as_posix(),
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            (Path(self.dataset_path) / 'valid').as_posix(),
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='categorical')

        return train_generator, validation_generator
