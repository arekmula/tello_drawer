import cv2
import numpy as np

from .helpers import crop_box
from .inference import HandDetector, HandClassifier


class ImageProcessor:
    def __init__(self, enlargebox_px=15, queue_size=20, drawing_state_threshold=0.5,
                 inactivity_std_dev_threshold=4, activity_std_dev_lower_threshold=15,
                 activity_std_dev_upper_threshold=100):
        self.hand_detector = HandDetector(confidence=0.6)
        self.hand_classifier = HandClassifier()

        self.enlargebox_pt = enlargebox_px
        self.drawing_state_threshold = drawing_state_threshold
        self.inactivity_std_dev_threshold = inactivity_std_dev_threshold
        self.activity_std_dev_lower_threshold = activity_std_dev_lower_threshold
        self.activity_std_dev_upper_threshold = activity_std_dev_upper_threshold

        self.image_size = self.hand_detector.get_image_size()
        self.path_image = np.zeros(shape=self.image_size, dtype=np.uint8)

        self.last_class_predictions = []
        self.last_box_predictions = []
        self.queue_size = queue_size

        self.drawing_state = False
        self.drawing_points = []
        self.finish_drawing = False
        self.is_outlier = False

    def process_img(self, frame):
        boxes, img_resized, image_resized_boxes = self.hand_detector.predict(img=frame, should_draw_results=True)
        boxes_images = crop_box(img_resized, boxes, enlargebox_px=self.enlargebox_pt)

        if len(boxes_images) > 0:
            if len(boxes_images) > 1:
                # TODO: Handle it better
                # If there's more than one hand, get right hand
                # Right hand has minimum x value
                # right_hand_index = np.argmin([box[0] for box in boxes])
                # boxes_images = [boxes_images[right_hand_index]]
                # boxes = [boxes[right_hand_index]]
                self.finish_drawing = True

            if not self.finish_drawing:
                for idx, (box_image, box) in enumerate(zip(boxes_images, boxes)):
                    prediction = self.hand_classifier.predict(box_image, should_preprocess_input=True)
                    box_middle = [int(box[0] + box[2]/2), int(box[1]+box[3]/2)]

                    self.add_predictions_to_queues(np.argmax(prediction), box_middle)
                    self.calculate_drawing_state()

                    if not self.is_outlier:
                        if self.drawing_state:
                            cv2.circle(self.path_image, tuple(box_middle), radius=2, color=(0, 255, 0), thickness=-1)
                            cv2.putText(self.path_image, str(idx), tuple(box_middle), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, color=(255, 0, 0))
                            self.drawing_points.append(box_middle)
                        else:
                            cv2.circle(self.path_image, tuple(box_middle), radius=2, color=(0, 0, 255), thickness=-1)
                        cv2.putText(image_resized_boxes, f"Drawing state: {self.drawing_state}", org=(0, 20),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))
                    else:
                        # Delete outliers
                        self.last_class_predictions.pop()
                        self.last_box_predictions.pop()
                        self.is_outlier = False

        if self.finish_drawing:
            self.normalize_drawing_points()

        return image_resized_boxes, self.path_image, self.finish_drawing, self.drawing_points

    def add_predictions_to_queues(self, class_prediction, box_prediction):
        if not(self.is_queue_full()):
            self.last_class_predictions.append(class_prediction)
            self.last_box_predictions.append(box_prediction)
        else:
            # Delete first element from queue
            self.last_class_predictions.pop(0)
            self.last_box_predictions.pop(0)
            # Add new element to queue
            self.last_class_predictions.append(class_prediction)
            self.last_box_predictions.append(box_prediction)

    def is_queue_full(self):
        if len(self.last_class_predictions) == self.queue_size and len(self.last_box_predictions) == self.queue_size:
            return True
        else:
            return False

    def calculate_drawing_state(self):
        """
        Calculates drawing state based on last predictions.

        self.last_class_predictions is a queue which stores last class predictions (0 for fist <stop signal> and 1
        for palm <start signal>). If the mean from queue is less than drawing_state_threshold and the hand didn't move
         it disables the drawing.

        :return:
        """
        if self.is_queue_full():
            # Compute mean of last class predictions. If closer to 0, then it's bigger chance that the Fist occured. If
            # closer to 1, then it's bigger chance that the Palm occured.
            class_predictions_mean = np.mean(self.last_class_predictions)

            # Compute standard deviation in last box positions
            last_box_predictions_array = np.array(self.last_box_predictions)
            box_std_dev = np.mean(np.std(last_box_predictions_array, axis=0))

            if (class_predictions_mean < self.drawing_state_threshold) \
                    and (box_std_dev < self.inactivity_std_dev_threshold):
                if self.drawing_state and not self.is_outlier:
                    # This is the case when fist is not moving, and previously state was drawing state
                    self.finish_drawing = True
                # This is the case when Fist is not moving
                self.drawing_state = False

            elif class_predictions_mean > self.drawing_state_threshold:
                if self.activity_std_dev_lower_threshold < box_std_dev < self.activity_std_dev_upper_threshold:
                    # This is the case when Palm is moving normally
                    self.drawing_state = True
                elif box_std_dev > self.activity_std_dev_upper_threshold:
                    # This is the case when Palm had a huge move. We classify this as a outlier
                    self.is_outlier = True

    def normalize_drawing_points(self):
        try:
            self.drawing_points = np.array(self.drawing_points, dtype=np.float)
            x_max = np.max(self.drawing_points[:, 0])
            self.drawing_points[:, 0] = self.drawing_points[:, 0] / x_max

            y_max = np.max(self.drawing_points[:, 1])
            self.drawing_points[:, 1] = self.drawing_points[:, 1] / y_max

        except IndexError as e:
            print("[HINT]: There was no points to normalize. Draw some shape before drawing stops")
            raise e
