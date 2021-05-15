import numpy as np
import cv2
from argparse import ArgumentParser

from inference import HandClassifier, HandDetector
from tello import Tello


def development_main(image_source, args, hand_detector, hand_classifier):
    if image_source == "built_camera":
        cap = cv2.VideoCapture(args.camera_index)
    else:
        cap = cv2.VideoCapture(args.filepath)

    while cap.isOpened():
        while True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                # Exit if q pressed
                cv2.destroyAllWindows()
                break

            ret, frame = cap.read()
            if ret:
                boxes, img_resize, image_resize_drawed = hand_detector.predict(img=frame, should_draw_results=True)
                hands = hand_detector.get_hand_from_img(img_resize, boxes, enlargebox_px=15)

                if len(hands) > 0:
                    for hand in hands:
                        prediction = hand_classifier.predict(hand, should_preprocess_input=True)
                        print("Palm" if np.argmax(prediction) else "Fist")

                cv2.imshow("frame", image_resize_drawed)
            else:
                break
        break

    cap.release()
    cv2.destroyAllWindows()


def tello_main(args):
    tello = Tello(local_ip=args.local_ip, local_port=args.local_port)


def main(args):
    hand_detector = HandDetector()
    hand_classifier = HandClassifier()

    image_source = args.image_source

    if image_source == "built_camera" or image_source == "saved_file":
        development_main(image_source=image_source, args=args,
                         hand_detector=hand_detector, hand_classifier=hand_classifier)
    else:
        tello_main(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--image_source", metavar="image_source", type=str, default="built_camera",
                        choices=["built_camera", "saved_file", "tello"])
    args, _ = parser.parse_known_args()
    if args.image_source == "saved_file":
        parser.add_argument("--filepath", metavar="filepath", type=str, required=True)
    elif args.image_source == "built_camera":
        parser.add_argument("--camera_index", metavar="camera_index", type=int, default=0)
    elif args.image_source == "tello":
        parser.add_argument("--local_ip", metavar="local_ip", type=str, default="0.0.0.0")
        parser.add_argument("--local_port", metavar="local_port", type=int, default=8889)

    args, _ = parser.parse_known_args()

    main(args)
