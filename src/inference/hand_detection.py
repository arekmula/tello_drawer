#!/usr/bin/env python3

import cv2
import sys
import tensorflow as tf
import numpy as np

from yolo import YOLO


class HandDetector:

    def __init__(self):
        self.yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
        self.yolo.size = int(416)
        self.yolo.confidence = float(0.2)

    def predict(self, img, draw):
        img_resize = cv2.resize(img, (416, 416))

        width, height, inference_time, results = self.yolo.inference(img_resize)

        conf_sum = 0
        detection_count = 0

        boxes = []

        for detection in results:
            id, name, confidence, x, y, w, h = detection
            boxes.append((x, y, w, h))

            if draw_bb:
                conf_sum += confidence
                detection_count += 1

                # draw a bounding box rectangle and label on the image
                color = (0, 0, 255)
                cv2.rectangle(img_resize, (x, y), (x + w, y + h), color, 3)
                text = "%s (%s)" % (name, round(confidence, 2))
                cv2.putText(img_resize, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, color, 1)

        if len(boxes) > 0:
            return boxes
        else:
            return None


if __name__ == '__main__':

    hand_detector = HandDetector()

    image = cv2.imread('./data/arek/circle/0000075.png')

    draw_bb = False

    b_boxes = hand_detector.predict(image, draw_bb)
    print(b_boxes)
