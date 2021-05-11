from hand_gesture_recognition import HandClassifier
from inference import HandDetector
from hand_gesture_recognition import HandClassifier
import argparse
import numpy as np
import cv2


def get_hand_from_img(image, boxes, enlargebox_px):
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

        bottom_right_x = x+w
        bottom_right_y = y+h

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


def main():
    print("main")

    hand_detector = HandDetector(image_size=416, confidence=0.2, model_dirs="models/")
    hand_classifier = HandClassifier(models_dir="models/")
    path_kuba = "../DATA/data/data_/kuba/VID_20210501_114442.mp4"
    path_maciej = "../DATA/data/data_/Helm_Thora/vokoscreen-2021-04-30_17-22-59.mkv"
    path_arek = "../DATA/data/data_/heart.avi"
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            boxes, img_resize, image_resize_drawed = hand_detector.predict(img=frame, should_draw_results=True)
            # TODO choose best hand
            hands = get_hand_from_img(img_resize, boxes, 15)

            if len(hands) > 0:
                for hand in hands:

                    hand_precessed = hand_classifier.preprocess_input(hand)
                    prediction = hand_classifier.predict(hand_precessed)

                    print("Palm" if np.argmax(prediction) else "Fist")

            cv2.imshow("frame", image_resize_drawed)
            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_data_path', type=str, help='Path to train images', required=True)
    # parser.add_argument('--valid_data_path', type=str, hekp='Path to valid images', required=True)
    # parser.add_argument('--test_data_path', type=str, help='Path to test images')
    main()
