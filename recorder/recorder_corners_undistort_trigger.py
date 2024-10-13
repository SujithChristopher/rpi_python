"""
this program records data from webcam and teensy controller
"""

import cv2
import os
import datetime
import msgpack as mp
import msgpack_numpy as mpn
import argparse
import time
from picamera2 import Picamera2
import libcamera
import gpiod
from cv2 import aruco
import sys
import keyboard
import toml
import numpy as np

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.useAruco3Detection = True

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
markerLength = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeperation,
    dictionary=ARUCO_DICT,
)
frame_size = (1280, 800)
WIDTH = frame_size[0]
HEIGHT = frame_size[1]

_fish_params = toml.load("/home/sujith/Documents/programs/undistort_best.toml")
# _fish_params = toml.load("undistort_best.toml")
_fish_matrix = np.array(_fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
_fish_dist = np.array(_fish_params["calibration"]["dist_coeffs"])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    _fish_matrix, _fish_dist, np.eye(3), _fish_matrix, (1200, 800), cv2.CV_16SC2
)


class RecordData:
    def __init__(
        self,
        _pth=None,
        record_camera=True,
        fps_value=30,
        isColor=False,
        default_res=False,
    ):
        self.picam2 = Picamera2()

        main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
        _c = {"FrameRate": 80, "ExposureTime": 3000}

        config = self.picam2.create_video_configuration(
            main, controls=_c, transform=libcamera.Transform(vflip=1)
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.record_camera = record_camera
        self.start_recording = False
        self._pth = _pth
        self.kill_signal = False
        self.fps_val = fps_value
        self.display = True

        self.isColor = isColor

        self.push_trigger = False
        self.push_frame_counter = 0
        self.recorded_signals = 0

        sync_pin = 17
        chip = gpiod.Chip("gpiochip4")
        self.sync_line = chip.get_line(sync_pin)
        self.sync_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)

    def capture_webcam(self):
        """capture webcam"""

        if self.record_camera:
            _save_pth = os.path.join(self._pth, "webcam_color.msgpack")
            _save_file = open(_save_pth, "wb")
            _timestamp_file = open(
                os.path.join(self._pth, "webcam_timestamp.msgpack"), "wb"
            )
        self.first_frame = True

        while True:
            frame = self.picam2.capture_array()
            gray_image = frame[:HEIGHT, :WIDTH]
            gray_image = cv2.flip(gray_image, 1)

            gray_image = cv2.remap(
                gray_image,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            corners, ids, rejected_image_points = detector.detectMarkers(gray_image)
            corners, ids, _, _ = detector.refineDetectedMarkers(
                gray_image, board, corners, ids, rejected_image_points
            )
            gray_image = aruco.drawDetectedMarkers(gray_image, corners, ids)
            # print(corners)
            # print(self.sync_line.get_value())
            if self.first_frame:
                _packed_file = mp.packb(gray_image.shape, default=mpn.encode)
                _save_file.write(_packed_file)
                self.first_frame = False

            if self.record_camera and self.start_recording:
                if ids is not None:
                    _packed_file = mp.packb([corners, ids], default=mpn.encode)
                    _save_file.write(_packed_file)
                else:
                    _packed_file = mp.packb([None, None], default=mpn.encode)
                    _save_file.write(_packed_file)

                _time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                if keyboard.is_pressed("t"):  # if key 't' is pressed
                    self.push_trigger = True

                if self.push_trigger:
                    self.push_frame_counter += 1
                    _packed_timestamp = mp.packb(
                        [
                            _time_stamp,
                            self.sync_line.get_value(),
                            str(self.recorded_signals),
                        ],
                        default=mpn.encode,
                    )
                    if self.push_frame_counter == 1:
                        print("triggered ", self.recorded_signals)

                    if self.push_frame_counter == 50:
                        self.push_trigger = False
                        self.push_frame_counter = 0
                        self.recorded_signals += 1
                        print("recorded")
                else:
                    _packed_timestamp = mp.packb(
                        [_time_stamp, self.sync_line.get_value(), str("NA")],
                        default=mpn.encode,
                    )

                _timestamp_file.write(_packed_timestamp)

            if self.display:
                gray_image_ = cv2.resize(gray_image, (250, 200))
                cv2.imshow("webcam", gray_image_)
                sys.stdout.flush()
                cv2.waitKey(1)

            if keyboard.is_pressed("s"):
                print("You Pressed a Key!, started recording from webcam")
                self.start_recording = True

            if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                if self.record_camera:
                    _save_file.close()
                    _timestamp_file.close()
                break

    def run(self):
        """run the program"""
        self.capture_webcam()


if __name__ == "__main__":
    """get parameter from external program"""

    parser = argparse.ArgumentParser(
        prog="Single camera recorder",
        description="This basically records data from the camera and the sensors",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--folder", help="folder name", required=False)
    parser.add_argument("-n", "--name", help="name of the file", required=False)
    parser.add_argument("-c", "--camera", help="record camera", required=False)
    parser.add_argument("-s", "--sensors", help="record sensors", required=False)

    args = parser.parse_args()

    # if your not passing any arguments then the default values will be used
    # and you may have to enter the folder name and the name of the recording
    if not any(vars(args).values()):
        print("No arguments passed, please enter manually")

        """Enter the respective parameters"""
        record_camera = True
        record_sensors = False

        if record_camera or record_sensors:
            _name = input("Enter the name of the recording: ")
        display = False
        _pth = None  # this is default do not change, path gets updated by your input
        _folder_name = (
            "recordings"  # this is the parent folder name where the data will be saved
        )

    else:
        print("Arguments passed")
        _folder_name = args.folder
        _name = args.name
        record_camera = args.camera
        record_sensors = args.sensors

        if record_camera == "True":
            record_camera = True
        else:
            record_camera = False
        if record_sensors == "True":
            record_sensors = True
        else:
            record_sensors = False

    if record_camera or record_sensors:
        _pth = os.path.join(
            os.path.dirname(__file__), "..", "data", _folder_name, _name
        )

        if "\n" in _pth:
            _pth = _pth.replace("\n", "")

        if not os.path.exists(_pth):
            os.makedirs(_pth)
    time.sleep(1)

    record_data = RecordData(
        _pth=_pth, record_camera=record_camera, isColor=True, default_res=True
    )
    record_data.run()
