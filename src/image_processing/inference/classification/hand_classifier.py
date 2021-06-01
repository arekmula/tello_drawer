#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import cv2


class HandClassifier:

    def __init__(self, models_dir='models/', model_name="trained_EfficientNetB0_palm_fist.h5"):
        try:
            self.model = tf.keras.models.load_model(models_dir + model_name)
        except OSError as e:
            raise e
        if self.model.name != "EfficientNet":
            self.input_shape = self.model.layers[0].input_shape
        else:
            self.input_shape = self.model.layers[0].input_shape[0]
        print("Classifier loaded successfully")

    def predict(self, input: np.array, should_preprocess_input) -> np.array:
        """
        :param input:
        :param should_preprocess_input:
        :return: array with class probabilities where 0 index is Fist and 1 index is Palm
        """
        if should_preprocess_input:
            input = self.preprocess_input(input)

        # This is basically to easier debugging
        assert input.shape == (1, self.input_shape[1], self.input_shape[2], self.input_shape[3]), \
            f"Incorrect input shape, should be {self.input_shape}"

        prediction = self.model.predict(input)
        
        return np.array(prediction)

    def preprocess_input(self, image) -> np.array:
        image = cv2.resize(image, (int(self.input_shape[1]), int(self.input_shape[2])))

        if self.model.name != "EfficientNet":
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image_grayscale / 255.
            image = image[:, :, np.newaxis]

        image = image[np.newaxis, :, :, :]
        return image
