#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import cv2


class HandClassifier:
    PREPROCESSED = False

    def __init__(self, models_dir='models/'):
        try:
            self.model = tf.keras.models.load_model(models_dir + "trained_palm_fist.h5")
        except OSError as e:
            raise e

        self.input_shape = self.model.layers[0].input_shape
        print("Classifier loaded successfully")

    def predict(self, input: np.array, should_preprocess_input) -> np.array:
        if should_preprocess_input:
            input = self.preprocess_input(input)

        # This is basically to easier debugging
        assert input.shape == (1, self.input_shape[1], self.input_shape[2], self.input_shape[3]), \
            f"Incorrect input shape, should be {self.input_shape}"

        prediction = self.model.predict(input)

        return np.array(prediction)

    def preprocess_input(self, image) -> np.array:
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_scaled = image_grayscale / 255.
        image_resized = cv2.resize(image_scaled, (int(self.input_shape[1]), int(self.input_shape[2])))
        image_final = image_resized[np.newaxis, :, :, np.newaxis]
        self.PREPROCESSED = True
        return image_final
