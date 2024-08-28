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
import gpiod


class RecordData:
    def __init__(
        self,
        _pth=None,
        record_camera=True,
        fps_value=30,
        isColor=False,
        default_res=False,
    ):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.record_camera = record_camera
        self.start_recording = False
        self._pth = _pth
        self.kill_signal = False
        self.fps_val = fps_value
        self.display = True

        self.isColor = isColor



    def capture_webcam(self):
        """capture webcam"""

        if self.record_camera:
            _save_pth = os.path.join(self._pth, "webcam_color.msgpack")
            _save_file = open(_save_pth, "wb")
            _timestamp_file = open(
                os.path.join(self._pth, "webcam_timestamp.msgpack"), "wb"
            )

        while self.capture.isOpened:
            ret, frame = self.capture.read()
            if ret:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                if self.record_camera and self.start_recording:
                    _packed_file = mp.packb(gray_image, default=mpn.encode)
                    _save_file.write(_packed_file)
                    _time_stamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )
                    _packed_timestamp = mp.packb(
                        [self.sync_line.get_value(), _time_stamp]
                    )
                    _timestamp_file.write(_packed_timestamp)
                    # print(self.sync_line.get_value())

                if self.display:
                    cv2.imshow("webcam", gray_image)
                    cv2.waitKey(1)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("You Pressed a Key!, ending webcam")
                    cv2.destroyAllWindows()
                    if self.record_camera:
                        _save_file.close()
                        _timestamp_file.close()
                    break

                if cv2.waitKey(1) & 0xFF == ord("s"):
                    print("You Pressed a Key!, started recording from webcam")
                    self.start_recording = True

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
