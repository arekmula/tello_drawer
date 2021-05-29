from argparse import ArgumentParser

import cv2
from djitellopy import Tello

from image_processing import ImageProcessor
from drone_steering import DroneSteering, distance, convert_to_distance_in_xy
import time
import numpy as np


def development_main(image_source, args):
    if image_source == "built_camera":
        cap = cv2.VideoCapture(args.camera_index)
    else:
        cap = cv2.VideoCapture(args.filepath)

    image_processor = ImageProcessor()

    while cap.isOpened():
        while True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                # Exit if q pressed
                cv2.destroyAllWindows()
                break

            ret, frame = cap.read()
            if ret:
                image_resize_drawed, path_img, finish_drawing, drawing_points = image_processor.process_img(frame)
                frame_and_path = cv2.hconcat([image_resize_drawed, path_img])
                if finish_drawing:
                    cv2.imshow("frame", frame_and_path)
                    key = cv2.waitKey(0)
                    break

                cv2.imshow("frame", frame_and_path)
            else:
                break
        break

    cap.release()
    cv2.destroyAllWindows()

    drone_steering = DroneSteering(max_area_cm=100)
    # speed_values = drone_steering.calculate_speed(drawing_points)
    rescaled_points = drone_steering.rescale_points(drawing_points)


def tello_main(args):
    image_processor = ImageProcessor()
    drone_steering = DroneSteering(max_area_cm=args.max_area)
    tello = drone_steering.get_tello_instance()

    drone_steering.start_pinging_tello()

    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            # Exit if q pressed
            cv2.destroyAllWindows()
            break

        frame = tello.get_frame_read().frame
        if frame is not None:
            image_resize_drawed, path_img, finish_drawing, drawing_points = image_processor.process_img(frame)
            frame_and_path = cv2.hconcat([image_resize_drawed, path_img])
            if finish_drawing:
                cv2.imshow("frame", frame_and_path)
                break
            cv2.imshow("frame", frame_and_path)

    drone_steering.stop_pinging_tello()

    rescaled_points = drone_steering.rescale_points(drawing_points)
    discrete_path = drone_steering.discrete_path(rescaled_points)
    discrete_path_distance = convert_to_distance_in_xy(discrete_path)
    drone_steering.reproduce_discrete_path_by_drone(discrete_path_distance)

    # Finish drawing
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(2)
    tello.land()


def main(args):
    image_source = args.image_source

    if image_source == "built_camera" or image_source == "saved_file":
        development_main(image_source=image_source, args=args)
    else:
        tello_main(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--image_source", metavar="image_source", type=str, default="tello",
                        choices=["built_camera", "saved_file", "tello"])
    args, _ = parser.parse_known_args()
    if args.image_source == "saved_file":
        parser.add_argument("--filepath", metavar="filepath", type=str, required=True)
    elif args.image_source == "built_camera":
        parser.add_argument("--camera_index", metavar="camera_index", type=int, default=0)
    elif args.image_source == "tello":
        parser.add_argument("--local_ip", metavar="local_ip", type=str, default="0.0.0.0")
        parser.add_argument("--local_port", metavar="local_port", type=int, default=8889)
        parser.add_argument("--max_area", metavar="max_area", type=int, default=100)

    args, _ = parser.parse_known_args()

    main(args)
