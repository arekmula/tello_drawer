import time

import numpy as np
from threading import Thread

from djitellopy import Tello
from .helpers import distance


class DroneProcessor:
    FINISH_DRAWING_HOLD_TIME_S = 2

    def __init__(self, max_area_cm=100, starting_move_up_cm=50, min_length_between_points_cm=5,
                 max_speed=30):
        """

        :param max_area_cm: Maximum length that drone can move from starting point in both axes.
        :param starting_move_up_cm: How many cms should drone go up after the takeoff
        :param min_length_between_points_cm: Minimum length between points, to reduce number of points from detection.
        """
        self.max_area = max_area_cm
        self.min_length_between_points_cm = min_length_between_points_cm
        self.max_speed = max_speed

        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.takeoff()
        self.tello.move_up(starting_move_up_cm)

        self.tello_ping_thread = Thread(target=self.ping_tello)
        self.should_stop_pinging_tello = False

    def get_last_frame(self):
        return self.tello.get_frame_read().frame

    def finish_drawing(self):
        """
        Finish drawing, by stopping drone in air for a while and then force it to land. Disable video streaming.

        :return:
        """
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(self.FINISH_DRAWING_HOLD_TIME_S)
        self.tello.land()
        self.tello.streamoff()

    def ping_tello(self):
        """
        Ping tello to prevent it from landing while drawing.

        :return:
        """
        while True:
            time.sleep(1)
            self.tello.send_command_with_return("command")
            print(f"Battery level: {self.tello.get_battery()}")
            if self.should_stop_pinging_tello:
                break

    def start_pinging_tello(self):
        """
        Starts thread that pings Tello drone, to prevent it from landing while drawing

        :return:
        """
        self.tello_ping_thread.start()

    def stop_pinging_tello(self):
        """
        Stop pinging tello to make it available to control

        :return:
        """
        self.should_stop_pinging_tello = True
        self.tello_ping_thread.join()

    def rescale_points(self, point_list, is_int=False):
        """
        Rescale points from 0-1 range to range defined by max_area.

        :param point_list:
        :param is_int:
        :return: Points rescaled to max_area
        """
        temp_list = []
        for point in point_list:
            temp_point = []
            for coordinate in point:
                coordinate = coordinate * self.max_area
                if is_int:
                    temp_point.append(int(coordinate))
                else:
                    temp_point.append(coordinate)
            temp_list.append(temp_point)
        return temp_list

    def discrete_path(self, rescaled_points):
        """
        Reduce number of points in list, so the difference between next points needs to be at least
        min_length_between_points_cm

        :param rescaled_points:
        :return:
        """
        last_index = -1
        length = 0
        while length < self.min_length_between_points_cm:
            last_index -= 1
            length = distance(rescaled_points[-1], rescaled_points[last_index])

        last_index = len(rescaled_points) + last_index
        discrete_path = [rescaled_points[0]]
        actual_point = 0
        for ind, point in enumerate(rescaled_points):
            if ind > last_index:
                discrete_path.append(rescaled_points[-1])
                break
            if distance(rescaled_points[actual_point], point) > 5:
                discrete_path.append(point)
                actual_point = ind

        return discrete_path

    def reproduce_discrete_path_by_drone(self, discrete_path):
        """
        Converts discrete path to velocity commands and sends them to drone, so the drone reproduce the path

        :param discrete_path: list of [x, y] points, which represents distance in each axis between previous point in
         list
        :return:
        """
        for current_point in discrete_path:
            ang = np.arctan2(current_point[0], current_point[1])
            x_speed = int(np.sin(ang) * self.max_speed)
            y_speed = -int(np.cos(ang) * self.max_speed)

            euclidean_distance = (current_point[0] ** 2 + current_point[1] ** 2) ** 0.5
            move_time = euclidean_distance / self.max_speed
            self.tello.send_rc_control(x_speed, 0, y_speed, 0)
            time.sleep(move_time)
