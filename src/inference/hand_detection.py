#!/usr/bin/env python3

import cv2
import numpy as np
from inference.yolo import YOLO


class HandDetector:

    def __init__(self, image_size=416, confidence=0.2, models_dir="models/"):
        try:
            self.yolo = YOLO(models_dir + "cross-hands-tiny.cfg", models_dir + "cross-hands-tiny.weights",
                             ["hand"])
        except OSError as e:
            raise e

        self.size = image_size
        self.confidence = confidence
        self.yolo.size = int(self.size)
        self.yolo.confidence = float(self.confidence)
        print("Detector loaded successfully")

    def predict(self, img, should_draw_results):
        """
        :param img: image
        :param should_draw_results:

        :return: parameters of bounding box
        x - x coordinate
        y - y coordinate
        w - box width
        h - box height
        """
        img_resize = cv2.resize(img, (self.size, self.size))
        image_resize_drawn = np.copy(img_resize)
        width, height, inference_time, results = self.yolo.inference(img_resize)

        conf_sum = 0
        detection_count = 0

        boxes = []

        for detection in results:
            id, name, confidence, x, y, w, h = detection
            boxes.append((x, y, w, h))

            if should_draw_results:
                conf_sum += confidence
                detection_count += 1

                # draw a bounding box rectangle and label on the image
                color = (0, 0, 255)
                cv2.rectangle(image_resize_drawn, (x, y), (x + w, y + h), color, 3)
                text = f"{name}, {round(confidence, 2)}"
                cv2.putText(image_resize_drawn, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, color, 1)

        return boxes, img_resize, image_resize_drawn

    def get_hand_from_img(self, image, boxes, enlargebox_px):
        hands = []

        for box in boxes:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # x, y, w, h = box
            # enlarge box a bit
            x -= enlargebox_px
            y -= enlargebox_px
            w += enlargebox_px * 2
            h += enlargebox_px * 2

            bottom_right_x = x + w
            bottom_right_y = y + h

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if bottom_right_x > image.shape[1]:
                bottom_right_x = image.shape[1]
            if bottom_right_y > image.shape[0]:
                bottom_right_y = image.shape[0]

            hand = image[y:bottom_right_y, x:bottom_right_x]
            hands.append(hand)

        return hands
