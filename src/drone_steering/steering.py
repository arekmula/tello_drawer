

class DroneSteering:
    def __init__(self, max_area=100, max_speed=10, signal_period=1):
        self.max_area = max_area
        self.max_speed = max_speed
        self.signal_period = signal_period

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


