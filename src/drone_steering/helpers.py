import numpy as np
import time


def convert_to_distance_in_xy(point_list):
    """
    Converts list of points in space to list of distances between previous point in list

    :param point_list:
    :return:
    """
    temp_list = []
    for i in range(1, len(point_list)):
        temp_list.append([point_list[i][0] - point_list[i - 1][0], point_list[i][1] - point_list[i - 1][1]])
    return temp_list


def convert_to_euclidean_distance(point_list):
    temp = []
    for i in range(1, len(point_list)):
        temp.append(distance(point_list[i], point_list[i - 1]))
    return temp


def distance(point1, point2):
    """
    Calculates distance between 2 points
    :param point1:
    :param point2:
    :return:
    """
    return np.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)


