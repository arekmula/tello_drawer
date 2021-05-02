import socket
import threading
import time
import numpy as np
import h264decoder


class Tello:
    """Wrapper class to interact with the Tello drone."""

    KPH_TO_CMS_MULTIPLIER = 27.7778
    MPH_TO_CMS_MULTIPLIER = 44.704
    POSSIBLE_TURN_DIRECTIONS = ['l', 'r', 'f', 'b']
    POSSIBLE_MOVE_DIRECTIONS = ["up", "down", "left", "right", "forward", "back"]
    FEETS_TO_CMS_MULTIPLIER = 30.48
    METERS_TO_CMS_MULTIPLIER = 100
    SINGLE_PACKET_MAX_BYTES = 1460

    def __init__(self, local_ip, local_port, imperial=False, command_timeout=.3, tello_ip='192.168.10.1',
                 tello_port=8889):
        """
        Binds to the local IP/port and puts the Tello into command mode.

        :param local_ip (str): Local IP address to bind.
        :param local_port (int): Local port to bind.
        :param imperial (bool): If True, speed is MPH and distance is feet.
                             If False, speed is KPH and distance is meters.
        :param command_timeout (int|float): Number of seconds to wait for a response to a command.
        :param tello_ip (str): Tello IP.
        :param tello_port (int): Tello port.
        """

        self.abort_flag = False
        self.decoder = h264decoder.H264Decoder()
        self.command_timeout = command_timeout
        self.imperial = imperial
        self.response = None
        self.frame = None  # numpy array BGR -- current camera output frame
        self.is_freeze = False  # freeze current camera output
        self.last_frame = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for receiving video stream
        self.tello_address = (tello_ip, tello_port)
        self.local_video_port = 11111  # port for receiving video stream
        self.last_height = 0
        self.socket.bind((local_ip, local_port))

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True

        self.receive_thread.start()

        # to receive video -- send cmd: command, streamon
        self.start_controlling_drone()
        self.set_video_stream(True)

        self.socket_video.bind((local_ip, self.local_video_port))

        # thread for receiving video
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True

        self.receive_video_thread.start()

    def __del__(self):
        """Closes the local socket."""

        self.socket.close()
        self.socket_video.close()

    def read(self):
        """Return the last frame from camera."""
        if self.is_freeze:
            return self.last_frame
        else:
            return self.frame

    def video_freeze(self, is_freeze=True):
        """Pause video output -- set is_freeze to True"""
        self.is_freeze = is_freeze
        if is_freeze:
            self.last_frame = self.frame

    def _receive_thread(self):
        """Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """
        while True:
            try:
                self.response, ip = self.socket.recvfrom(3000)
            except socket.error as exc:
                print(f"Caught exception socket.error : {exc}")

    def _receive_video_thread(self):
        """
        Listens for video streaming (raw h264) from the Tello.

        Runs as a thread, sets self.frame to the most recent frame Tello captured.

        """
        packet_data = b""
        while True:
            try:
                res_string, ip = self.socket_video.recvfrom(2048)
                packet_data += res_string
                # end of frame
                if len(res_string) != self.SINGLE_PACKET_MAX_BYTES:
                    for frame in self._h264_decode(packet_data):
                        self.frame = frame
                    packet_data = b""

            except socket.error as exc:
                print(f"Caught exception socket.error: {exc}")

    def _h264_decode(self, packet_data):
        """
        decode raw h264 format data from Tello

        :param packet_data: raw h264 data array

        :return: a list of decoded frame
        """
        res_frame_list = []
        frames = self.decoder.decode(packet_data)
        for framedata in frames:
            (frame, w, h, ls) = framedata
            if frame is not None:
                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((h, int(ls / 3), 3)))
                frame = frame[:, :w, :]
                res_frame_list.append(frame)

        return res_frame_list

    def send_command(self, command) -> str:
        """
        Send a command to the Tello and wait for a response.

        :param command: Command to send.
        :return (str): Response from Tello.

        """

        print(f">> send cmd: {command}")
        self.abort_flag = False
        timer = threading.Timer(self.command_timeout, self.set_abort_flag)

        self.socket.sendto(command.encode(), self.tello_address)

        timer.start()
        while self.response is None:
            if self.abort_flag is True:
                break
        timer.cancel()

        if self.response is None:
            response = 'none_response'
        else:
            response = self.response.decode()

        self.response = None

        return response

    def start_controlling_drone(self) -> str:
        """
        Enter SDK mode.

        :return: Response from Tello
        """
        return self.send_command("command")

    def set_video_stream(self, state: bool) -> str:
        """
        Turn on/off streaming tello video stream.

        :param state: if True, then Tello will stream the video. If False then video will not be streamed.
        :return: Response from Tello
        """
        if state:
            return self.send_command("streamon")
        else:
            return self.send_command("streamoff")

    def set_abort_flag(self):
        """
        Sets self.abort_flag to True.

        Used by the timer in Tello.send_command() to indicate to that a response

        timeout has occurred.

        """

        self.abort_flag = True

    def takeoff(self) -> str:
        """
        Initiates take-off.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command('takeoff')

    def set_speed(self, speed: float) -> str:
        """
        Sets speed.

        This method expects KPH or MPH. The Tello API expects speeds from
        1 to 100 centimeters/second.

        Metric: .1 to 3.6 KPH
        Imperial: .1 to 2.2 MPH

        Args:
            speed (int|float): Speed.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        speed = float(speed)

        if self.imperial is True:
            speed = int(round(speed * self.MPH_TO_CMS_MULTIPLIER))
        else:
            speed = int(round(speed * self.KPH_TO_CMS_MULTIPLIER))

        return self.send_command(f'speed {speed}')

    def rotate_cw(self, degrees: int) -> str:
        """
        Rotates clockwise.

        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command(f'cw {degrees}')

    def rotate_ccw(self, degrees: int) -> str:
        """
        Rotates counter-clockwise.
        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.send_command(f'ccw {degrees}')

    def flip(self, direction: str):
        """
        Flips.

        Args:
            direction (str): Direction to flip, 'l', 'r', 'f', 'b'.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        if direction not in self.POSSIBLE_TURN_DIRECTIONS:
            print(f"Input direction {direction} is not possible. Possible directions: {self.POSSIBLE_TURN_DIRECTIONS}")
            return "FALSE"

        return self.send_command('flip %s' % direction)

    def get_response(self) -> int:
        """
        Returns response of tello.

        Returns:
            int: response of tello.

        """
        response = self.response
        return response

    def get_height(self) -> int:
        """Returns height(dm) of tello.

        Returns:
            int: Height(dm) of tello.

        """
        # TODO: Not sure if this command is still available in SDK 2.0

        height = self.send_command('height?')
        height = str(height)
        height = filter(str.isdigit, height)
        try:
            height = int(height)
            self.last_height = height
        except:
            height = self.last_height
            pass
        return height

    def get_battery(self) -> int:
        """Returns percent battery life remaining.

        Returns:
            int: Percent battery life remaining.

        """

        battery = self.send_command('battery?')

        try:
            battery = int(battery)
        except:
            pass

        return battery

    def get_flight_time(self) -> int:
        """Returns the number of seconds elapsed during flight.

        Returns:
            int: Seconds elapsed during flight.

        """

        flight_time = self.send_command('time?')

        try:
            flight_time = int(flight_time)
        except:
            pass

        return flight_time

    def get_speed(self) -> float:
        """Returns the current speed.

        Returns:
            float: Current speed in KPH or MPH.

        """

        speed = self.send_command('speed?')

        try:
            speed = float(speed)

            if self.imperial is True:
                speed = round((speed / self.MPH_TO_CMS_MULTIPLIER), 1)
            else:
                speed = round((speed / self.KPH_TO_CMS_MULTIPLIER), 1)
        except:
            pass

        return speed

    def land(self) -> str:
        """Initiates landing.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command('land')

    def move(self, direction: str, distance) -> str:
        """Moves in a direction for a distance.

        This method expects meters or feet. The Tello API expects distances
        from 20 to 500 centimeters.

        Metric: .02 to 5 meters
        Imperial: .7 to 16.4 feet

        Args:
            direction (str): Direction to move, 'forward', 'back', 'right' or 'left'.
            distance (int|float): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        if direction not in self.POSSIBLE_MOVE_DIRECTIONS:
            print(f"Input direction {direction} is not possible. Possible directions: {self.POSSIBLE_MOVE_DIRECTIONS}")
            return "FALSE"
        
        distance = float(distance)

        if self.imperial is True:
            distance = int(round(distance * self.FEETS_TO_CMS_MULTIPLIER))
        else:
            distance = int(round(distance * self.METERS_TO_CMS_MULTIPLIER))

        return self.send_command(f'{direction} {distance}')

    def move_backward(self, distance) -> str:
        """Moves backward for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('back', distance)

    def move_down(self, distance) -> str:
        """Moves down for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('down', distance)

    def move_forward(self, distance) -> str:
        """Moves forward for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.move('forward', distance)

    def move_left(self, distance) -> str:
        """Moves left for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.move('left', distance)

    def move_right(self, distance) -> str:
        """Moves right for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        """
        return self.move('right', distance)

    def move_up(self, distance) -> str:
        """Moves up for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('up', distance)

    def go_xyz_speed(self, x: int, y: int, z: int, speed: int):
        """Fly to x y z relative to the current position.
        Speed defines the traveling speed in cm/s.
        Arguments:
            x: 20-500
            y: 20-500
            z: 20-500
            speed: 10-100
        """
        cmd = f'go {x} {y} {z} {speed}'
        self.send_control_command(cmd)
