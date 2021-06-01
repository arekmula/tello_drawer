from argparse import ArgumentParser

import cv2

from image_processing import ImageProcessor
from drone_processing import DroneProcessor, convert_to_distance_in_xy


def development_main(image_source, args):
    """
    Main function used to development using built-in camera or file.

    :param image_source:
    :param args:
    :return:
    """
    if image_source == "built_camera":
        cap = cv2.VideoCapture(args.camera_index)
    else:
        cap = cv2.VideoCapture(args.filepath)

    image_processor = ImageProcessor(finish_drawing_sign=args.finish_drawing,
                                     hand_detector_confidence=args.hand_detection_confidence)

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


def tello_main(args):
    """
    Main function used to control your drone using hand.

    :param args:
    :return:
    """
    image_processor = ImageProcessor(finish_drawing_sign=args.finish_drawing,
                                     hand_detector_confidence=args.hand_detection_confidence)
    drone_processor = DroneProcessor(max_area_cm=args.max_area, min_length_between_points_cm=args.min_length,
                                     starting_move_up_cm=args.takeoff_offset)

    # Start pinigng tello to prevent it from landing
    drone_processor.start_pinging_tello()

    # Drawing loop
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            # Exit if q pressed
            cv2.destroyAllWindows()
            break

        frame = drone_processor.get_last_frame()
        if frame is not None:
            image_resize_drawed, path_img, finish_drawing, drawing_points = image_processor.process_img(frame)
            frame_and_path = cv2.hconcat([image_resize_drawed, path_img])
            if finish_drawing:
                cv2.imshow("frame", frame_and_path)
                break
            cv2.imshow("frame", frame_and_path)

    # Stop pinging, so we can send move commands to the drone
    drone_processor.stop_pinging_tello()
    # Rescale points from range 0-1 to range defined by max_area.
    rescaled_points = drone_processor.rescale_points(drawing_points)
    # Reduce number of points to reproduce
    discrete_path = drone_processor.discrete_path(rescaled_points)
    # Convert point list, to list of differences between previous point in list
    discrete_path_distance = convert_to_distance_in_xy(discrete_path)
    # Reproduce path by the drone
    drone_processor.reproduce_discrete_path_by_drone(discrete_path_distance)
    # Finish drawing
    drone_processor.finish_drawing()

    cv2.destroyAllWindows()


def main(args):
    image_source = args.image_source

    print(f"Image source: {args.image_source}")
    print(f"Finish drawing sign: {args.finish_drawing}")
    print(f"Hand detection confidence: {args.hand_detection_confidence}")

    if image_source == "built_camera" or image_source == "saved_file":
        development_main(image_source=image_source, args=args)
    else:
        tello_main(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--image_source", metavar="image_source", type=str, default="tello",
                        choices=["built_camera", "saved_file", "tello"])
    parser.add_argument("--finish_drawing", metavar="finish_drawing", type=str, default="two_hands",
                        choices=["two_hands", "fist"], help="Finish drawing sign")
    args, _ = parser.parse_known_args()
    if args.image_source == "saved_file":
        parser.add_argument("--filepath", metavar="filepath", type=str, required=True)
    elif args.image_source == "built_camera":
        parser.add_argument("--camera_index", metavar="camera_index", type=int, default=0)
    elif args.image_source == "tello":
        parser.add_argument("--max_area", metavar="max_area", type=int, default=100,
                            help="The max area [cm] that drone can use to perform the drawing")
        parser.add_argument("--min_length", metavar="min_length", type=int, default=5,
                            help="Minimum length between points, to reduce number of points from detection")
        parser.add_argument("--takeoff_offset", metavar="takeoff_offset", type=int, default=50,
                            help="Takeoff move up offset in cm.")

    parser.add_argument("--hand_detection_confidence", metavar="hand_detection_confidence",
                        type=float, default=0.6 if args.finish_drawing == "fist" else 0.85,
                        help="The confidence for hand detector should be lower, because we have to detect fist also."
                             "For two hands detector the confidence has to be higher to get rid of false positives.")

    args, _ = parser.parse_known_args()

    main(args)
