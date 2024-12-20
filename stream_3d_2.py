import numpy as np
import cv2
from cv2 import aruco
import socket
import toml
import os

try:
    import libcamera
    from picamera2 import Picamera2

    WEBCAM = False
except:
    WEBCAM = True


ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.useAruco3Detection = 1
ARUCO_PARAMETERS.cornerRefinementMethod = 2
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
markerLength = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeperation,
    dictionary=ARUCO_DICT,
)

frame_size = (1200, 800)


def estimate_pose_single_markers(
    corners, marker_size, camera_matrix, distortion_coefficients
):
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    rvecs = []
    tvecs = []
    for corner in corners:
        _, r, t = cv2.solvePnP(
            marker_points,
            corner,
            camera_matrix,
            distortion_coefficients,
            True,
            # flags=cv2.SOLVEPNP_IPPE_SQUARE,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if r is not None and t is not None:
            r = np.array(r).reshape(1, 3).tolist()
            t = np.array(t).reshape(1, 3).tolist()
            rvecs.append(r)
            tvecs.append(t)
    return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)


_fish_params = toml.load("/home/sujith/Documents/programs/undistort_best.toml")
# _fish_params = toml.load("undistort_best.toml")
_fish_matrix = np.array(_fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
_fish_dist = np.array(_fish_params["calibration"]["dist_coeffs"])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    _fish_matrix, _fish_dist, np.eye(3), _fish_matrix, (1200, 800), cv2.CV_16SC2
)


class ExponentialMovingAverageFilter3D:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema_x = None
        self.ema_y = None
        self.ema_z = None

    def update(self, ema):
        if self.ema_x is None:
            self.ema_x = ema[0]
            self.ema_y = ema[1]
            self.ema_z = ema[2]
        else:
            self.ema_x = self.alpha * ema[0] + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * ema[1] + (1 - self.alpha) * self.ema_y
            self.ema_z = self.alpha * ema[2] + (1 - self.alpha) * self.ema_z
        return np.array([self.ema_x, self.ema_y, self.ema_z])


class MainClass:
    def __init__(self, cam_calib_path, udp_stream=False):
        self.ar_pos = None
        self.UDP_STREAM = udp_stream

        if not WEBCAM:
            self.picam2 = Picamera2()
            WIDTH = frame_size[0]
            HEIGHT = frame_size[1]
            main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
            _c = {"FrameRate": 100, "ExposureTime": 5000}
            config = self.picam2.create_video_configuration(
                main, controls=_c, transform=libcamera.Transform(vflip=1)
            )
            self.picam2.configure(config)
            self.picam2.start()
        else:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.rvec = None
        self.tvec = None
        self.first_rvec = None
        self.first_tvec = None
        self.close_file = False

        self.FIRST_FRAME = True
        cam_calib_path = cam_calib_path

        self.default_ids = [4, 8, 12, 14, 20]
        self.received_message = ""
        self.stop_signal = False

        self.RMAT_INIT = False
        self.initial_rmat = np.eye(3)

        self.tvec_04 = np.nan
        self.tvec_08 = np.nan
        self.tvec_12 = np.nan
        self.tvec_24 = np.nan
        self.tvec_20 = np.nan

        self.rvec_04 = np.nan
        self.rvec_08 = np.nan
        self.rvec_12 = np.nan
        self.rvec_14 = np.nan
        self.rvec_20 = np.nan

        self.tv_origin = np.zeros((3, 1))

        self.does_not_exist = []

        self.zero_vec = np.zeros(3)
        self.tvec_dist = np.zeros(3)
        self.temp_tvec = np.zeros((3, 1))
        self.rmat = np.eye(3)
        self.save_first_frame = False

        # self.data_file = open(self.save_data_path, 'wb')
        # self.raw_data_file = open(self.raw_data, 'wb')
        self.raw_data_trigger = False

        data = toml.load(cam_calib_path)

        self.camera_matrix = np.array(data["calibration"]["camera_matrix"]).reshape(
            3, 3
        )

        self.distortion_coeff = np.array(data["calibration"]["dist_coeffs"])

        # Example usage
        alpha = 0.4  # Smoothing factor between 0 and 1
        self.filter = ExponentialMovingAverageFilter3D(alpha=alpha)

        if self.UDP_STREAM:
            udp_ip = "localhost"
            udp_port = 8000
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind((udp_ip, udp_port))
            self.udp_socket.settimeout(5)
            print(self.udp_socket.getsockname())

    def preprocess_ids(self, ids, rotation_vectors, translation_vectors):
        self.does_not_exist = []
        for idx, id in enumerate(ids):
            match np.array(id):
                case 4:
                    self.tvec_04 = translation_vectors[idx][0]
                    self.rvec_04 = rotation_vectors[idx][0]
                case 8:
                    self.tvec_08 = translation_vectors[idx][0]
                    self.rvec_08 = rotation_vectors[idx][0]
                case 12:
                    self.tvec_12 = translation_vectors[idx][0]
                    self.rvec_12 = rotation_vectors[idx][0]
                case 14:
                    self.tvec_14 = translation_vectors[idx][0]
                    self.rvec_14 = rotation_vectors[idx][0]
                case 20:
                    self.tvec_20 = translation_vectors[idx][0]
                    self.rvec_20 = rotation_vectors[idx][0]

        _ids = np.array(ids)
        if self.default_ids[0] not in _ids:
            self.does_not_exist.append(self.default_ids[0])
        if self.default_ids[1] not in _ids:
            self.does_not_exist.append(self.default_ids[1])
        if self.default_ids[2] not in _ids:
            self.does_not_exist.append(self.default_ids[2])
        if self.default_ids[3] not in _ids:
            self.does_not_exist.append(self.default_ids[3])
        if self.default_ids[4] not in _ids:
            self.does_not_exist.append(self.default_ids[4])

        for _d in self.does_not_exist:
            match np.array(_d):
                case 4:
                    self.tvec_04 = np.nan
                    self.rvec_04 = np.nan
                case 8:
                    self.tvec_08 = np.nan
                    self.rvec_08 = np.nan
                case 12:
                    self.tvec_12 = np.nan
                    self.rvec_12 = np.nan
                case 14:
                    self.tvec_14 = np.nan
                    self.rvec_14 = np.nan
                case 20:
                    self.tvec_20 = np.nan
                    self.rvec_20 = np.nan

    def coordinate_transform(self):
        tv_04 = np.nan
        tv_08 = np.nan
        tv_12 = np.nan
        tv_14 = np.nan
        tv_20 = np.nan

        rm_04 = np.eye(3)
        rm_08 = np.eye(3)
        rm_12 = np.eye(3)
        rm_14 = np.eye(3)
        rm_20 = np.eye(3)

        id_04_offset = np.array([0.00, 0.1, -0.069]).reshape(3, 1)
        id_08_offset = np.array([0.00, 0.01, -0.069]).reshape(3, 1)
        id_12_offset = np.array([0.00, 0.0, -0.1075]).reshape(3, 1)
        id_14_offset = np.array([-0.09, 0.0, -0.069]).reshape(3, 1)
        id_20_offset = np.array([0.1, 0.0, -0.069]).reshape(3, 1)

        if self.tvec_04 is not np.nan:
            tv_04 = np.array(self.tvec_04).reshape(3, 1)
            rm_04 = cv2.Rodrigues(self.rvec_04)[0]
        if self.tvec_08 is not np.nan:
            tv_08 = np.array(self.tvec_08).reshape(3, 1)
            rm_08 = cv2.Rodrigues(self.rvec_08)[0]
        if self.tvec_12 is not np.nan:
            tv_12 = np.array(self.tvec_12).reshape(3, 1)
            rm_12 = cv2.Rodrigues(self.rvec_12)[0]
        if self.tvec_14 is not np.nan:
            tv_14 = np.array(self.tvec_14).reshape(3, 1)
            rm_14 = cv2.Rodrigues(self.rvec_14)[0]
        if self.tvec_20 is not np.nan:
            tv_20 = np.array(self.tvec_20).reshape(3, 1)
            rm_20 = cv2.Rodrigues(self.rvec_20)[0]

        pa_c_04 = rm_04 @ id_04_offset + tv_04
        pa_b_c_04 = self.initial_rmat.T @ (pa_c_04 - self.tv_origin)

        pa_c_08 = rm_08 @ id_08_offset + tv_08
        pa_b_c_08 = self.initial_rmat.T @ (pa_c_08 - self.tv_origin)

        pa_c_12 = rm_12 @ id_12_offset + tv_12
        pa_b_c_12 = self.initial_rmat.T @ (pa_c_12 - self.tv_origin)

        pa_c_14 = rm_14 @ id_14_offset + tv_14
        pa_b_c_14 = self.initial_rmat.T @ (pa_c_14 - self.tv_origin)

        pa_c_20 = rm_20 @ id_20_offset + tv_20
        pa_b_c_20 = self.initial_rmat.T @ (pa_c_20 - self.tv_origin)

        _rmat = cv2.Rodrigues(self.first_rvec)[0]

        _tr_04 = _rmat.T @ (pa_b_c_04 - self.first_tvec.reshape(3, 1))
        _tr_08 = _rmat.T @ (pa_b_c_08 - self.first_tvec.reshape(3, 1))
        _tr_12 = _rmat.T @ (pa_b_c_12 - self.first_tvec.reshape(3, 1))
        _tr_14 = _rmat.T @ (pa_b_c_14 - self.first_tvec.reshape(3, 1))
        _tr_20 = _rmat.T @ (pa_b_c_20 - self.first_tvec.reshape(3, 1))

        self.tvec_dist = np.nanmedian(
            np.array(
                [
                    _tr_04,
                    _tr_08,
                    _tr_12,
                    _tr_14,
                    _tr_20,
                ]
            ),
            axis=0,
        )

        # print(self.tvec_dist)
        self.tvec_dist = self.tvec_dist.T[0]

    def camera_thread(self):
        WIDTH = frame_size[0]
        HEIGHT = frame_size[1]
        self.previous_vec = np.zeros(3)
        self.current_vec = np.zeros(3)
        self.first_vec = True

        while True:
            if self.UDP_STREAM:
                self.received_message, self.addr = self.udp_socket.recvfrom(30)

                if self.received_message.decode("utf-8") == "stop":
                    self.udp_socket.sendto("stop".encode("utf-8"), self.addr)
                    self.udp_socket.close()
                    break

                if self.received_message.decode("utf-8") == "close":
                    self.udp_socket.sendto("close".encode("utf-8"), self.addr)
                    self.data_file.close()
                    self.close_file = True
                    self.raw_data_trigger = True

                if self.received_message.decode("utf-8") == "set_orgin":
                    self.save_first_frame = True
                    self.udp_socket.sendto("saved".encode("utf-8"), self.addr)
            if not WEBCAM:
                self.video_frame = self.picam2.capture_array()
                self.video_frame = self.video_frame[:HEIGHT, :WIDTH]
                self.video_frame = cv2.flip(self.video_frame, 1)
                self.video_frame = cv2.remap(
                    self.video_frame,
                    map1,
                    map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            else:
                _, self.video_frame = self.camera.read()

            if self.FIRST_FRAME:
                corners, ids, rejectedpoints = detector.detectMarkers(self.video_frame)
                corners, ids, rejectedpoints, _ = detector.refineDetectedMarkers(
                    image=self.video_frame,
                    board=board,
                    detectedCorners=corners,
                    detectedIds=ids,
                    rejectedCorners=rejectedpoints,
                    # cameraMatrix=self.camera_matrix,
                    # distCoeffs=self.distortion_coeff,
                )
                # print(corners, self.video_frame.shape)
                self.video_frame = aruco.drawDetectedMarkers(
                    self.video_frame, corners, ids
                )
                if ids is not None:
                    self.first_rvec, self.first_tvec = estimate_pose_single_markers(
                        corners, 0.05, self.camera_matrix, self.distortion_coeff
                    )
                    self.first_rvec = self.first_rvec[0]
                    self.first_tvec = self.first_tvec[0]
                    self.FIRST_FRAME = False
                # self.coordinate_transform()
            else:
                # gray = cv2.cvtColor(self.video_frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejectedpoints = detector.detectMarkers(self.video_frame)
                corners, ids, rejectedpoints, _ = detector.refineDetectedMarkers(
                    image=self.video_frame,
                    board=board,
                    detectedCorners=corners,
                    detectedIds=ids,
                    rejectedCorners=rejectedpoints,
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.distortion_coeff,
                )
                self.video_frame = aruco.drawDetectedMarkers(
                    self.video_frame, corners, ids
                )
                if (ids is not None and len(ids) > 0) and all(
                    item in self.default_ids for item in np.array(ids)
                ):
                    self.rvec, self.tvec = estimate_pose_single_markers(
                        corners, 0.05, self.camera_matrix, self.distortion_coeff
                    )

                    for _r, _t in zip(self.rvec, self.tvec):
                        cv2.drawFrameAxes(
                            self.video_frame,
                            self.camera_matrix,
                            self.distortion_coeff,
                            _r,
                            _t,
                            0.05,
                        )

                    self.preprocess_ids(ids, self.rvec, self.tvec)
                    self.coordinate_transform()

                    self.tvec_dist = self.filter.update(self.tvec_dist)

                    # Check if this is the first vector
                    if self.first_vec:
                        self.previous_vec = (
                            self.tvec_dist * 100
                        )  # Initialize previous vector
                        self.current_vec = (
                            self.previous_vec
                        )  # Initialize current vector
                        self.first_vec = False  # Update the first_vec flag
                    else:
                        self.current_vec = (
                            self.tvec_dist * 100
                        )  # Update the current vector
                        norm = np.linalg.norm(self.current_vec - self.previous_vec)

                        # Set a threshold for jitter reduction
                        if (
                            norm > 10
                        ):  # Adjust this threshold based on the specific context
                            self.tvec_cm = (
                                self.previous_vec
                            )  # Use the previous vector if the change is too large

                        else:
                            self.tvec_cm = (
                                self.current_vec
                            )  # Use the current vector otherwise
                            self.previous_vec = self.current_vec

                        self.tvec_cm = self.current_vec
                        self.tvec_x = (
                            str(-1 * self.tvec_cm[0])
                            + ","
                            + str(-1 * self.tvec_cm[1])
                            + ","
                            + str(-1 * self.tvec_cm[2])
                        )

                        if self.UDP_STREAM:
                            self.udp_socket.sendto(
                                str(self.tvec_x).encode("utf-8"), self.addr
                            )

                        self.save_first_frame = False
                        if self.UDP_STREAM:
                            self.udp_socket.sendto(
                                str(self.tvec_x).encode("utf-8"), self.addr
                            )
                        else:
                            if self.UDP_STREAM:
                                self.udp_socket.sendto(
                                    "none".encode("utf-8"), self.addr
                                )

            self.video_frame = cv2.resize(self.video_frame, (350, 200))
            cv2.imshow("frame", self.video_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def run(self):
        self.camera_thread()


if __name__ == "__main__":
    _file_path = "/home/sujith/Documents/programs/calib_undistort_aruco.toml"
    _file_path = "/home/sujith/Documents/programs/calib_mono_faith3D.toml"

    # _file_path = 'calib_undistort_aruco.toml'
    print(_file_path)

    """
    Check these parameters
    """
    UDP_STREAM = True
    CAMERA_CALIBRATION_FILE = _file_path

    """
    Then run the main program
    """

    main = MainClass(cam_calib_path=CAMERA_CALIBRATION_FILE, udp_stream=UDP_STREAM)
    main.run()
