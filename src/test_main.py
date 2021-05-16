from inference import HandDetector
from hand_gesture_recognition import HandClassifier
import numpy as np
import cv2


def main():
    hand_detector = HandDetector()
    hand_classifier = HandClassifier()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            boxes, img_resize, image_resize_drawed = hand_detector.predict(img=frame, should_draw_results=True)
            # TODO choose best hand
            hands = hand_detector.get_hand_from_img(img_resize, boxes, enlargebox_px = 15)

            if len(hands) > 0:
                for hand in hands:

                    prediction = hand_classifier.predict(hand, should_preprocess_input=True)

                    print("Palm" if np.argmax(prediction) else "Fist")

            cv2.imshow("frame", image_resize_drawed)
            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
