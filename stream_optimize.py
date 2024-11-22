import numpy as np
import cv2
from cv2 import aruco
import platform
import socket
import toml
import os
from filters import ExponentialMovingAverageFilter3D


class Config:
    FRAME_SIZE = (1200, 800)
    MARKER_LENGTH = 0.05
    MARKER_SEPARATION = 0.01
    UDP_IP = "localhost"
    UDP_PORT = 8000
    DEFAULT_IDS = [4, 8, 12, 14, 20]
    ALPHA = 0.4  # Exponential moving average filter smoothing factor


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
        self.camera_matrix = np.array(calib_data["calibration"]["camera_matrix"]).reshape(3, 3)
        self.distortion_coeff = np.array(calib_data["calibration"]["dist_coeffs"])

        self.detector = self._init_detector()
        self.board = self._init_board()

        self.picam2, self.map1, self.map2 = None, None, None
        self.video_frame = None
        self.tvec_dist = np.zeros(3)
        self.first_frame = True

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
        fish_params = toml.load("/home/sujith/Documents/programs/undistort_best.toml")
        fish_matrix = np.array(fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
        fish_dist = np.array(fish_params["calibration"]["dist_coeffs"])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            fish_matrix, fish_dist, np.eye(3), fish_matrix, self.frame_size, cv2.CV_16SC2
        )

    def _init_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def _init_udp_socket(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((Config.UDP_IP, Config.UDP_PORT))
        self.udp_socket.settimeout(5)
        print("UDP socket initialized:", self.udp_socket.getsockname())

    def estimate_pose(self, corners):
        marker_points = np.array([
            [-self.marker_length / 2, self.marker_length / 2, 0],
            [self.marker_length / 2, self.marker_length / 2, 0],
            [self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0],
        ], dtype=np.float32)

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

    def process_frame(self):
        # Capture frame
        ret = None
        if platform.system() == "Linux":
            self.video_frame = self.picam2.capture_array()
            self.video_frame = cv2.remap(
                self.video_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
            )
        else:
            ret, self.video_frame = self.camera.read()
            self.video_frame = cv2.flip(self.video_frame, 1)
        
        if self.video_frame is None or ret is False:
            return

        corners, ids, _ = self.detector.detectMarkers(self.video_frame)
        if ids is not None:
            self.video_frame = aruco.drawDetectedMarkers(self.video_frame, corners, ids)
            rvecs, tvecs = self.estimate_pose(corners)
            self._draw_axes(rvecs, tvecs)

        self.video_frame = cv2.resize(self.video_frame, (350, 200))
        cv2.imshow("frame", self.video_frame)

    def _draw_axes(self, rvecs, tvecs):
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(self.video_frame, self.camera_matrix, self.distortion_coeff, rvec, tvec, 0.05)

    def run(self):
        while True:
            self.process_frame()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CAMERA_CALIB_PATH = "/home/sujith/Documents/programs/calib_mono_faith3D.toml" 
    CAMERA_CALIB_PATH = r"E:\CMC\pyprojects\programs_rpi\rpi_python\calib_mono_1200_800.toml" 
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH, udp_stream=False)
    main.run()
