import cv2
import numpy as np
from .helpers import crop_box
from .inference import HandDetector, HandClassifier


class ImageProcessor:
    def __init__(self, enlargebox_px=15):
        self.hand_detector = HandDetector()
        self.hand_classifier = HandClassifier()

        self.enlargebox_pt = enlargebox_px
        self.image_size = self.hand_detector.get_image_size()
        self.path_image = np.zeros(shape=self.image_size, dtype=np.uint8)

    def process_img(self, frame):
        boxes, img_resized, image_resized_boxes = self.hand_detector.predict(img=frame, should_draw_results=True)
        boxes_images = crop_box(img_resized, boxes, enlargebox_px=self.enlargebox_pt)

        if len(boxes_images) > 0:
            if len(boxes_images) > 1: # If there's more than one hand, get right hand
                right_hand_index = np.argmin([box[0] for box in boxes])
                boxes_images = [boxes_images[right_hand_index]]
                boxes = [boxes[right_hand_index]]

            for box_image, box in zip(boxes_images, boxes):
                prediction = self.hand_classifier.predict(box_image, should_preprocess_input=True)

                box_middle = (box[0], box[1])

                cv2.circle(self.path_image, box_middle, radius=0, color=(255, 0, 0), thickness=-1)

                print("Palm" if np.argmax(prediction) else "Fist")

        return image_resized_boxes, self.path_image
