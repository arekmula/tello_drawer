from argparse import ArgumentParser
from threading import Thread

import cv2
from djitellopy import Tello

from image_processing import ImageProcessor
from drone_steering import DroneSteering, distance, convert_to_distance_in_xy
import time
import numpy as np

stop_thread_value = False

V=30

def ping_tello(tello_object):
    while True:
        time.sleep(1)
        global stop_thread_value
        tello_object.send_command_with_return("command")
        print(tello_object.get_battery())
        if stop_thread_value:
            break


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

    drone_steering = DroneSteering(max_area=100, max_speed=100, signal_period=1)
    # speed_values = drone_steering.calculate_speed(drawing_points)
    rescaled_points = drone_steering.rescale_points(drawing_points)

    last_index = -1
    length = 0
    while length < 5.0:
        last_index -= 1
        length = distance(rescaled_points[-1], rescaled_points[last_index])

    last_index = len(rescaled_points) + last_index
    move_list = [rescaled_points[0]]
    actual_point = 0
    for ind, point in enumerate(rescaled_points):
        if ind > last_index:
            move_list.append(rescaled_points[-1])
            break
        if distance(rescaled_points[actual_point], point) > 5:
            move_list.append(point)
            actual_point = ind

    move_list_distance = convert_to_distance_in_xy(move_list)

    for move in move_list_distance:
        ang = np.arctan2(move[0], move[1])
        x_speed = int(np.sin(ang) * V)
        y_speed = int(np.cos(ang) * V)

        c = (move[0] ** 2 + move[1] ** 2) ** 0.5
        move_time = c / V
        time.sleep(move_time)


def tello_main(args):
    image_processor = ImageProcessor()
    tello = Tello()
    tello.connect()
    tello.streamon()
    tello.takeoff()
    tello.move_up(50)

    global stop_thread_value
    stop_thread_value = False
    tello_ping_thread = Thread(target=ping_tello, args=[tello])
    tello_ping_thread.start()

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
                # key = cv2.waitKey(0)
                break
            cv2.imshow("frame", frame_and_path)

    # cv2.destroyAllWindows()
    stop_thread_value = True
    tello_ping_thread.join()

    drone_steering = DroneSteering(max_area=200, max_speed=100, signal_period=1)
    # speed_values = drone_steering.calculate_speed(drawing_points)
    rescaled_points = drone_steering.rescale_points(drawing_points)

    last_index = -1
    length = 0
    while length < 5.0:
        last_index -= 1
        length = distance(rescaled_points[-1], rescaled_points[last_index])

    last_index = len(rescaled_points) + last_index
    move_list = [rescaled_points[0]]
    actual_point = 0
    for ind, point in enumerate(rescaled_points):
        if ind > last_index:
            move_list.append(rescaled_points[-1])
            break
        if distance(rescaled_points[actual_point], point) > 5:
            move_list.append(point)
            actual_point = ind

    move_list_distance = convert_to_distance_in_xy(move_list)

    for move in move_list_distance:
        ang = np.arctan2(move[0], move[1])
        x_speed = int(np.sin(ang) * V)
        y_speed = -int(np.cos(ang) * V)

        c = (move[0] ** 2 + move[1] ** 2) ** 0.5
        move_time = c / V
        tello.send_rc_control(x_speed, 0, y_speed, 0)
        time.sleep(move_time)

    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(2)

    # for speed_value in speed_values:
    #     time.sleep(0.1)
    #     print(speed_value)
    #     tello.send_rc_control(speed_value[0], 0, speed_value[1], 0)

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

    args, _ = parser.parse_known_args()

    main(args)
