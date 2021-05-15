import cv2
import numpy as np
from queue import Queue

from .helpers import crop_box
from .inference import HandDetector, HandClassifier


class ImageProcessor:
    def __init__(self, enlargebox_px=15, drawing_state_threshold=0.01, queue_size=50):
        self.hand_detector = HandDetector()
        self.hand_classifier = HandClassifier()

        self.enlargebox_pt = enlargebox_px
        self.drawing_state_threshold = drawing_state_threshold

        self.image_size = self.hand_detector.get_image_size()
        self.path_image = np.zeros(shape=self.image_size, dtype=np.uint8)

        self.last_class_predictions = Queue(queue_size)
        self.last_box_predictions = Queue(queue_size)

        self.drawing_state = False
        self.drawing_points = []
        self.stopped_drawing = False

    def process_img(self, frame):
        boxes, img_resized, image_resized_boxes = self.hand_detector.predict(img=frame, should_draw_results=True)
        boxes_images = crop_box(img_resized, boxes, enlargebox_px=self.enlargebox_pt)

        if len(boxes_images) > 0:
            if len(boxes_images) > 1:
                # TODO: Handle it better
                # If there's more than one hand, get right hand
                # Right hand has minimum x value
                right_hand_index = np.argmin([box[0] for box in boxes])
                boxes_images = [boxes_images[right_hand_index]]
                boxes = [boxes[right_hand_index]]

            for box_image, box in zip(boxes_images, boxes):
                prediction = self.hand_classifier.predict(box_image, should_preprocess_input=True)
                box_middle = [box[0], box[1]]

                self.add_predictions_to_queues(np.argmax(prediction), box_middle)
                self.calculate_drawing_state()

                if self.drawing_state:
                    cv2.circle(self.path_image, tuple(box_middle), radius=2, color=(0, 255, 0), thickness=-1)
                    self.drawing_points.append(box_middle)
                else:
                    cv2.circle(self.path_image, tuple(box_middle), radius=2, color=(0, 0, 255), thickness=-1)
                cv2.putText(image_resized_boxes, f"Drawing state: {self.drawing_state}", org=(0, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))

        return image_resized_boxes, self.path_image, self.stopped_drawing, self.drawing_points

    def add_predictions_to_queues(self, class_prediction, box_prediction):
        if not(self.last_class_predictions.full()):
            self.last_class_predictions.put(class_prediction)
        else:
            self.last_class_predictions.get()
            self.last_class_predictions.put(class_prediction)

        if not (self.last_box_predictions.full()):
            self.last_box_predictions.put(box_prediction)
        else:
            self.last_box_predictions.get()
            self.last_box_predictions.put(box_prediction)

    def calculate_drawing_state(self):
        """
        Calculates drawing state based on last predictions.

        self.last_class_predictions is a queue which stores last class predictions (0 for fist <stop signal> and 1
        for palm <start signal>). If the mean from queue is less than drawing_state_threshold it disables the drawing.

        :return:
        """
        if self.last_class_predictions.full():
            class_predictions_mean = np.mean([item for item in self.last_class_predictions.queue])
            if class_predictions_mean < self.drawing_state_threshold:
                if self.drawing_state:
                    self.stopped_drawing = True
                self.drawing_state = False
            else:
                self.drawing_state = True
