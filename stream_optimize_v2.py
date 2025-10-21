import numpy as np
import cv2
from cv2 import aruco
import platform
import socket
import toml
import os
from filters import ExponentialMovingAverageFilter3D
import struct
import csv
from datetime import datetime


class Config:
    FRAME_SIZE = (1200, 800)
    MARKER_LENGTH = 0.05
    MARKER_SEPARATION = 0.01
    UDP_IP = "localhost"
    UDP_PORT = 8000
    DEFAULT_IDS = [4, 8, 12, 14, 20]
    ALPHA = 0.4  # Exponential moving average filter smoothing factor
    MARKER_OFFSETS = {
        4: np.array([0.00, 0.1, -0.069]),
        8: np.array([0.00, 0.01, -0.069]),
        12: np.array([0.00, 0.0, -0.1075]),
        14: np.array([-0.09, 0.0, -0.069]),
        20: np.array([0.1, 0.0, -0.069]),
    }


class StreamState:
    """Enum-like class for tracking stream state"""
    IDLE = 0
    REFERENCE_CAPTURED = 1
    TRACKING = 2
    RECORDING = 3


class MainClass:
    def __init__(self, cam_calib_path, udp_stream=False):
        self.udp_stream = udp_stream
        self.filter = ExponentialMovingAverageFilter3D(alpha=Config.ALPHA)
        self.default_ids = Config.DEFAULT_IDS
        self.frame_size = Config.FRAME_SIZE
        self.marker_length = Config.MARKER_LENGTH
        self.marker_separation = Config.MARKER_SEPARATION

        # Load calibration parameters
        calib_data = toml.load(cam_calib_path)
        self.camera_matrix = np.array(
            calib_data["calibration"]["camera_matrix"]
        ).reshape(3, 3)
        self.distortion_coeff = np.array(calib_data["calibration"]["dist_coeffs"])

        self.detector = self._init_detector()
        self.board = self._init_board()

        self.picam2, self.map1, self.map2 = None, None, None
        self.video_frame = None
        self.tvec_dist = np.zeros(3)

        # Enhanced state management
        self.state = StreamState.IDLE
        self.reference_captured = False
        self.first_id = None
        self.first_rvec = None
        self.first_tvec = None

        self.save_path = None
        self.csv_writer = None
        self.csv_file = None
        self.record = False

        self.received_message = ""
        self.addr = None

        self._curr_session = os.path.join('Session-' + datetime.today().strftime('%Y-%m-%d'), 'MovementData')

        if platform.system() == "Linux":
            self._init_rpi_camera()
        else:
            self._init_camera()

        if self.udp_stream:
            self._init_udp_socket()

    def _init_detector(self):
        aruco_params = aruco.DetectorParameters()
        aruco_params.useAruco3Detection = True
        aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        return aruco.ArucoDetector(aruco_dict, aruco_params)

    def _init_board(self):
        return aruco.GridBoard(
            size=(1, 1),
            markerLength=self.marker_length,
            markerSeparation=self.marker_separation,
            dictionary=self.detector.getDictionary(),
        )

    def _init_rpi_camera(self):
        from picamera2 import Picamera2
        import libcamera

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            {"format": "YUV420", "size": self.frame_size},
            controls={"FrameRate": 100, "ExposureTime": 5000},
            transform=libcamera.Transform(vflip=1),
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Load fisheye calibration
        fish_params = toml.load("/home/sujith/Documents/rpi_python/undistort_best.toml")
        fish_matrix = np.array(fish_params["calibration"]["camera_matrix"]).reshape(
            3, 3
        )
        fish_dist = np.array(fish_params["calibration"]["dist_coeffs"])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            fish_matrix,
            fish_dist,
            np.eye(3),
            fish_matrix,
            self.frame_size,
            cv2.CV_16SC2,
        )

    def _init_camera(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def _init_udp_socket(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((Config.UDP_IP, Config.UDP_PORT))
        self.udp_socket.setblocking(False)
        print(f"UDP socket initialized: {self.udp_socket.getsockname()}")
        print("\n=== UDP Command Reference ===")
        print("CAPTURE_REF    - Capture current pose as reference frame")
        print("RESET_REF      - Clear reference frame and return to IDLE")
        print("START_TRACK    - Start tracking (requires reference frame)")
        print("STOP_TRACK     - Stop tracking")
        print("USER:<id>      - Set hospital ID and start recording")
        print("CHANGE:<id>    - Change hospital ID")
        print("STOP           - Stop everything and exit")
        print("STATUS         - Request current state")
        print("=============================\n")

    def estimate_pose(self, corners):
        marker_points = np.array(
            [
                [-self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, -self.marker_length / 2, 0],
                [-self.marker_length / 2, -self.marker_length / 2, 0],
            ],
            dtype=np.float32,
        )

        rvecs, tvecs = [], []
        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                corner,
                self.camera_matrix,
                self.distortion_coeff,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                rvecs.append(rvec.flatten())
                tvecs.append(tvec.flatten())
        return np.array(rvecs), np.array(tvecs)

    def _draw_axes(self, rvecs, tvecs):
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(
                self.video_frame,
                self.camera_matrix,
                self.distortion_coeff,
                rvec,
                tvec,
                0.05,
            )

    def _get_centroid(self, ids, rvecs, tvecs):
        ids = np.array(ids).flatten()
        tvecs = np.array(tvecs).reshape(len(ids), 3)
        rvecs = np.array(rvecs).reshape(len(ids), 3)

        _transformed = np.full((len(ids), 3), np.nan)
        for index, _id in enumerate(ids):
            if _id in Config.MARKER_OFFSETS:
                _transformed[index] = (
                    cv2.Rodrigues(rvecs[index])[0]
                    @ Config.MARKER_OFFSETS[_id].reshape(3, 1)
                    + tvecs[index].reshape(3, 1)
                ).T[0]

        return np.nanmean(_transformed, axis=0).flatten()

    def _get_local_coordinates(self, first_id, first_rvecs, first_tvecs, centroid):
        first_id = np.array(first_id).flatten()
        first_tvecs = np.array(first_tvecs).reshape(len(first_id), 3)
        first_rvecs = np.array(first_rvecs).reshape(len(first_id), 3)

        _id = first_id[0]
        _r = cv2.Rodrigues(first_rvecs[0])[0]
        _t = first_tvecs[0]

        _local_camera_t = _r @ Config.MARKER_OFFSETS[_id].reshape(3, 1) + _t.reshape(
            3, 1
        )

        _local_camera_t = _local_camera_t.T[0]
        _local_coordinates = _r.T @ (_local_camera_t - centroid).reshape(3, 1)
        return _local_coordinates.T[0]

    def _send_message(self, message_type, data=None):
        """Send structured messages to Godot"""
        if not self.udp_stream or self.addr is None:
            return

        if data is None:
            data = np.zeros(3)

        # Message type codes
        message_codes = {
            "IDLE": 0.0,
            "REFERENCE_CAPTURED": 1.0,
            "TRACKING": 2.0,
            "RECORDING": 3.0,
            "ERROR": -1.0,
            "STOP": -99.0,
        }

        _msg = message_codes.get(message_type, 0.0)
        _data = np.append(_msg, data).flatten()
        _data_bytes = struct.pack("f" * len(_data), *_data)
        self.udp_socket.sendto(_data_bytes, self.addr)

    def _capture_reference_frame(self, ids, rvecs, tvecs):
        """Capture current frame as reference for coordinate system"""
        if ids is not None and len(ids) > 0:
            self.first_id = ids
            self.first_rvec = rvecs
            self.first_tvec = tvecs
            self.reference_captured = True
            self.state = StreamState.REFERENCE_CAPTURED
            print(f"Reference frame captured with markers: {ids.flatten()}")
            self._send_message("REFERENCE_CAPTURED")
            return True
        else:
            print("ERROR: No markers detected, cannot capture reference frame")
            self._send_message("ERROR")
            return False

    def _reset_reference_frame(self):
        """Clear reference frame and return to IDLE state"""
        self.first_id = None
        self.first_rvec = None
        self.first_tvec = None
        self.reference_captured = False
        self.state = StreamState.IDLE
        self.filter = ExponentialMovingAverageFilter3D(alpha=Config.ALPHA)  # Reset filter
        print("Reference frame reset")
        self._send_message("IDLE")

    def _handle_udp_commands(self, ids, rvecs, tvecs):
        """Process UDP commands from Godot"""
        if not hasattr(self, 'received_message') or not self.received_message:
            return None

        message = self.received_message
        self.received_message = ""  # Clear message after processing

        try:
            if message == b"CAPTURE_REF":
                return self._capture_reference_frame(ids, rvecs, tvecs)

            elif message == b"RESET_REF":
                self._reset_reference_frame()
                return None

            elif message == b"START_TRACK":
                if self.reference_captured:
                    self.state = StreamState.TRACKING
                    print("Tracking started")
                    self._send_message("TRACKING")
                else:
                    print("ERROR: Cannot start tracking without reference frame")
                    self._send_message("ERROR")
                return None

            elif message == b"STOP_TRACK":
                if self.record:
                    self._stop_recording()
                self.state = StreamState.REFERENCE_CAPTURED
                print("Tracking stopped")
                self._send_message("REFERENCE_CAPTURED")
                return None

            elif message.startswith(b"USER:"):
                if self.reference_captured:
                    self._hid = message.decode("utf-8").split(":")[1]
                    if self.save_path is None:
                        self._select_hospitalid()
                    self.state = StreamState.RECORDING
                    self.record = True
                    print(f"Recording started for user: {self._hid}")
                    self._send_message("RECORDING")
                else:
                    print("ERROR: Cannot start recording without reference frame")
                    self._send_message("ERROR")
                return None

            elif message.startswith(b"CHANGE:"):
                if self.reference_captured:
                    self._stop_recording()
                    self._hid = message.decode("utf-8").split(":")[1]
                    self._select_hospitalid()
                    self.state = StreamState.RECORDING
                    self.record = True
                    print(f"Changed to user: {self._hid}")
                    self._send_message("RECORDING")
                else:
                    print("ERROR: Cannot change user without reference frame")
                    self._send_message("ERROR")
                return None

            elif message == b"STATUS":
                state_names = {
                    StreamState.IDLE: "IDLE",
                    StreamState.REFERENCE_CAPTURED: "REFERENCE_CAPTURED",
                    StreamState.TRACKING: "TRACKING",
                    StreamState.RECORDING: "RECORDING"
                }
                current_state = state_names.get(self.state, "UNKNOWN")
                print(f"Current state: {current_state}, Reference: {self.reference_captured}")
                self._send_message(current_state)
                return None

            elif message == b"STOP":
                print("Received STOP command")
                self._send_message("STOP")
                return "STOP"

        except Exception as e:
            print(f"Error handling command: {e}")
            self._send_message("ERROR")

        return None

    def _stop_recording(self):
        """Stop recording and close CSV file"""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        self.record = False
        self.save_path = None
        print("Recording stopped and file closed")

    def _draw_status_overlay(self):
        """Draw status information on video frame"""
        state_text = {
            StreamState.IDLE: "IDLE - Waiting for CAPTURE_REF",
            StreamState.REFERENCE_CAPTURED: "REF CAPTURED - Ready to track",
            StreamState.TRACKING: "TRACKING",
            StreamState.RECORDING: "RECORDING"
        }

        text = state_text.get(self.state, "UNKNOWN")
        color = (0, 255, 0) if self.reference_captured else (0, 0, 255)

        # Add semi-transparent background
        overlay = self.video_frame.copy()
        cv2.rectangle(overlay, (5, 5), (345, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, self.video_frame, 0.7, 0, self.video_frame)

        cv2.putText(self.video_frame, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_frame(self):
        # Capture frame
        ret = None
        if platform.system() == "Linux":
            self.video_frame = self.picam2.capture_array()
            self.video_frame = cv2.remap(
                self.video_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
            )
            self.video_frame = cv2.flip(self.video_frame, 1)
        else:
            ret, self.video_frame = self.camera.read()

        if self.video_frame is None or ret is False:
            return None

        # Receive UDP messages
        if self.udp_stream:
            try:
                self.received_message, self.addr = self.udp_socket.recvfrom(30)
            except socket.error:
                pass

        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(self.video_frame)

        if ids is not None:
            self.video_frame = aruco.drawDetectedMarkers(self.video_frame, corners, ids)
            rvecs, tvecs = self.estimate_pose(corners)
            self._draw_axes(rvecs, tvecs)

            # Handle UDP commands
            command_result = self._handle_udp_commands(ids, rvecs, tvecs)
            if command_result == "STOP":
                return "STOP"

            # Only compute and send coordinates if we have a reference frame
            if self.reference_captured and self.state in [StreamState.TRACKING, StreamState.RECORDING]:
                _centroid = self._get_centroid(ids, rvecs, tvecs)
                _local_coordinates = self._get_local_coordinates(
                    self.first_id, self.first_rvec, self.first_tvec, _centroid
                )
                _local_coordinates = self.filter.update(_local_coordinates)

                # Send coordinates to Godot
                state_msg = "RECORDING" if self.state == StreamState.RECORDING else "TRACKING"
                self._send_message(state_msg, _local_coordinates)

                # Record to CSV if in recording state
                if self.record and self.csv_writer is not None:
                    self.csv_writer.writerow([datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                             *_local_coordinates])

        else:
            # No markers detected, still handle commands
            self._handle_udp_commands(None, None, None)

        # Draw status overlay
        self._draw_status_overlay()

        # Resize and display
        self.video_frame = cv2.resize(self.video_frame, (350, 200))
        cv2.imshow("frame", self.video_frame)

        return None

    def _select_hospitalid(self):
        """Create directory and CSV file for recording"""
        if self.save_path is None:
            self.save_path = os.path.join(
                os.path.expanduser("~/Documents/NOARK/data"), self._hid, self._curr_session
            )
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            filename = os.path.join(
                self.save_path,
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_data.csv"
            )
            self.csv_file = open(filename, "w", newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["Time", "X", "Y", "Z"])
            print(f"Recording to: {filename}")

    def run(self):
        import time
        last_heartbeat = time.time()

        print("Starting stream...")
        print("Waiting for commands from Godot...")

        while True:
            try:
                result = self.process_frame()

                # Update heartbeat if we received a message
                if hasattr(self, 'received_message') and self.received_message:
                    last_heartbeat = time.time()

                # Check for STOP command
                if result == "STOP":
                    print("Stopping stream...")
                    break

                # Heartbeat timeout (only if UDP is enabled)
                if self.udp_stream and time.time() - last_heartbeat > 5.0:
                    print("Lost connection to Godot, exiting...")
                    break

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received, exiting...")
                break
            except Exception as e:
                print(f"Error in process_frame: {e}")
                import traceback
                traceback.print_exc()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("'q' pressed, exiting...")
                break

        # Cleanup
        if self.csv_file is not None:
            self.csv_file.close()
        cv2.destroyAllWindows()
        print("Stream stopped")


if __name__ == "__main__":
    if platform.system() == "Linux":
        CAMERA_CALIB_PATH = "/home/sujith/Documents/rpi_python/old_calibration/calib_mono_faith3D.toml"
    else:
        CAMERA_CALIB_PATH = (
            r"E:\CMC\pyprojects\programs_rpi\rpi_python\webcam_calib.toml"
        )
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH, udp_stream=True)
    main.run()
