from .helpers import convert_to_distance_in_xy
import numpy as np

class DroneSteering:
    def __init__(self, max_area=100, max_speed=10, signal_period=1):
        self.max_area = max_area
        self.max_speed = max_speed
        self.signal_period = signal_period
        self.max_distance = self.signal_period * self.max_speed

    def rescale_points(self, point_list, is_int=False):
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

    def add_extra_points(self, dist_list):
        temp_list = []
        for dist in dist_list:
            dst = np.sqrt(dist[0]**2 + dist[1]**2)
            if dst <= self.max_distance:
                temp_list.append(dist)
            else:
                ang = np.arctan2(dist[0], dist[1])
                maximum_move = [self.max_distance*np.sin(ang), self.max_distance * np.cos(ang)]
                last_point = dist
                for i in range(int(dst / self.max_distance)):
                    temp_list.append(maximum_move)
                    last_point[0] -= maximum_move[0]
                    last_point[1] -= maximum_move[1]
                temp_list.append(last_point)

        return temp_list

    def calculate_speed(self, point_list):
        point_list = self.rescale_points(point_list, is_int=False)
        distances_list = convert_to_distance_in_xy(point_list)
        # distances_list = self.add_extra_points(distances_list)
        speed_list = []
        for distance in distances_list:
            speed = [3 * int(distance[0] / self.signal_period), 3 * int(distance[1] / self.signal_period)]
            speed_list.append(speed)

        return speed_list


