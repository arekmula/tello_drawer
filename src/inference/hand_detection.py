#!/usr/bin/env python3

import cv2

from yolo import YOLO


class HandDetector:

    def __init__(self, image_size=416, confidence=0.2):
        self.yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
        self.size = image_size
        self.confidence = confidence
        self.yolo.size = int(self.size)
        self.yolo.confidence = float(self.confidence)

    def predict(self, img, should_draw_results):
        """
        :param img: image
        :param should_draw_results:

        :return: parameters of bounding box
        x - x coordinate
        y - y coordinate
        w - box width
        h - boz height
        """
        img_resize = cv2.resize(img, (self.size, self.size))

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
                cv2.rectangle(img_resize, (x, y), (x + w, y + h), color, 3)
                text = f"{name}, {round(confidence, 2)}"
                cv2.putText(img_resize, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, color, 1)

        return boxes, img_resize
