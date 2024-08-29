import numpy as np
import cv2
from cv2 import aruco
from picamera2 import Picamera2
import libcamera
import socket
import toml
import os

# ArUco Marker and Board Parameters
ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.useAruco3Detection = 1
ARUCO_PARAMETERS.cornerRefinementMethod = 3
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

markerLength = 0.05
markerSeparation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeparation,
    dictionary=ARUCO_DICT,
)

frame_size = (1200, 800)

def estimate_pose_single_markers(corners, marker_size, camera_matrix, distortion_coefficients):
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
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if r is not None and t is not None:
            rvecs.append(r.reshape(1, 3).tolist())
            tvecs.append(t.reshape(1, 3).tolist())
    return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)

# Camera Calibration
_fish_params = toml.load('/home/sujith/Documents/programs/undistort_best.toml')
_fish_matrix = np.array(_fish_params['calibration']['camera_matrix']).reshape(3, 3)
_fish_dist = np.array(_fish_params['calibration']['dist_coeffs'])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    _fish_matrix, _fish_dist, np.eye(3), _fish_matrix, frame_size, cv2.CV_16SC2
)

class ExponentialMovingAverageFilter3D:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema = np.zeros(3)

    def update(self, value):
        self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

class MainClass:
    def __init__(self, cam_calib_path, udp_stream=False):
        self.UDP_STREAM = udp_stream
        self.picam2 = Picamera2()

        # Configure the camera
        config = self.picam2.create_video_configuration(
            {"format": "YUV420", "size": frame_size},
            controls={"FrameRate": 100, 'ExposureTime':1000},
            transform=libcamera.Transform(vflip=1)
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Initialization
        self.FIRST_FRAME = True
        self.default_ids = [12, 88, 89, 14, 20]
        self.does_not_exist = []
        self.raw_data_trigger = False
        self.save_first_frame = False

        data = toml.load(cam_calib_path)
        self.camera_matrix = np.array(data["calibration"]["camera_matrix"]).reshape(3, 3)
        self.distortion_coeff = np.array(data["calibration"]["dist_coeffs"])

        # EMA Filter
        self.filter = ExponentialMovingAverageFilter3D(alpha=0.4)

        if self.UDP_STREAM:
            udp_ip = "localhost"
            udp_port = 8000
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind((udp_ip, udp_port))
            self.udp_socket.settimeout(5)
            print(self.udp_socket.getsockname())

    def preprocess_ids(self, ids, rotation_vectors, translation_vectors):
        self.does_not_exist = [id for id in self.default_ids if id not in ids]

        # Reset to NaN for missing IDs
        self.tvec_dict = {id: np.nan for id in self.default_ids}
        self.rvec_dict = {id: np.nan for id in self.default_ids}

        for idx, id in enumerate(ids):
            if id in self.default_ids:
                self.tvec_dict[str(id[0])] = translation_vectors[idx][0]
                self.rvec_dict[str(id[0])] = rotation_vectors[idx][0]

    def coordinate_transform(self):
        offsets = {
            12: np.array([0.00, 0.0, -0.1075]),
            88: np.array([0.00, 0.0, -0.1075]),
            89: np.array([0.1, 0.0, -0.055]),
            14: np.array([-0.09, 0.0, -0.069]),
            20: np.array([0.1, 0.0, -0.069]),
        }

        transformed_tvecs = []
        for id, offset in offsets.items():
            tvec = self.tvec_dict.get(id)
            rvec = self.rvec_dict.get(id)
            if not np.isnan(tvec):
                rm = cv2.Rodrigues(rvec)[0]
                pa_c = rm @ offset.reshape(3, 1) + np.array(tvec).reshape(3, 1)
                pa_b_c = self.initial_rmat.T @ (pa_c - self.tv_origin)
                transformed_tvecs.append(pa_b_c)

        _rmat = cv2.Rodrigues(self.first_rvec)[0] if self.first_rvec is not None else np.eye(3)
        self.tvec_dist = np.nanmedian([self.initial_rmat.T @ (pa_b_c - self.first_tvec.reshape(3, 1)) for pa_b_c in transformed_tvecs], axis=0).flatten()
        print(self.tvec_dist)

    def camera_thread(self):
        previous_vec = np.zeros(3)
        current_vec = np.zeros(3)
        first_vec = True

        while True:
            if self.UDP_STREAM:
                try:
                    self.received_message, self.addr = self.udp_socket.recvfrom(30)
                    message = self.received_message.decode("utf-8")

                    if message == "stop":
                        self.udp_socket.sendto("stop".encode("utf-8"), self.addr)
                        self.udp_socket.close()
                        break
                    elif message == "close":
                        self.udp_socket.sendto("close".encode("utf-8"), self.addr)
                        self.close_file = True
                        self.raw_data_trigger = True
                    elif message == "set_orgin":
                        self.save_first_frame = True
                        self.udp_socket.sendto("saved".encode("utf-8"), self.addr)
                except socket.timeout:
                    continue

            self.video_frame = self.picam2.capture_array()
            self.video_frame = cv2.flip(self.video_frame[:frame_size[1], :frame_size[0]], 1)
            self.video_frame = cv2.remap(self.video_frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            if self.FIRST_FRAME:
                corners, ids, rejected_corners = detector.detectMarkers(self.video_frame)
                corners, ids, _, _ = detector.refineDetectedMarkers(
                    self.video_frame, board, detectedCorners=corners, detectedIds=ids, rejectedCorners=rejected_corners
                )
                self.video_frame = aruco.drawDetectedMarkers(self.video_frame, corners, ids)
                if ids is not None:
                    self.first_rvec, self.first_tvec = estimate_pose_single_markers(
                        corners, markerLength, self.camera_matrix, self.distortion_coeff
                    )
                    self.FIRST_FRAME = False
            else:
                corners, ids, rejected_corners = detector.detectMarkers(self.video_frame)
                corners, ids, _, _ = detector.refineDetectedMarkers(
                    self.video_frame, board, detectedCorners=corners, detectedIds=ids,
                    cameraMatrix=self.camera_matrix, distCoeffs=self.distortion_coeff, rejectedCorners=rejected_corners
                )
                self.video_frame = aruco.drawDetectedMarkers(self.video_frame, corners, ids)
                if ids is not None and len(ids) > 0 and all(id in self.default_ids for id in ids):
                    self.rvec, self.tvec = estimate_pose_single_markers(
                        corners, markerLength, self.camera_matrix, self.distortion_coeff
                    )
                    for _r, _t in zip(self.rvec, self.tvec):
                        cv2.drawFrameAxes(self.video_frame, self.camera_matrix, self.distortion_coeff, _r, _t, markerLength / 2)

                    self.preprocess_ids(ids, self.rvec, self.tvec)
                    self.coordinate_transform()
                    self.tvec_dist = self.filter.update(self.tvec_dist)

                    if first_vec:
                        previous_vec = current_vec = self.tvec_dist * 100
                        first_vec = False
                    else:
                        current_vec = self.tvec_dist * 100
                        norm = np.linalg.norm(current_vec - previous_vec)
                        self.tvec_cm = previous_vec if norm > 10 else current_vec
                        previous_vec = current_vec

                    if self.UDP_STREAM:
                        try:
                            data_to_send = f"{self.tvec_cm[0]},{self.tvec_cm[1]},{self.tvec_cm[2]}"
                            self.udp_socket.sendto(data_to_send.encode('utf-8'), self.addr)
                        except Exception as e:
                            print(f"Error sending UDP data: {e}")

            resized_frame = cv2.resize(self.video_frame, (350, 200))
            cv2.imshow("frame", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    cam_calib_path = '/home/sujith/Documents/programs/undistort_best.toml'
    main_class_instance = MainClass(cam_calib_path, udp_stream=False)
    main_class_instance.camera_thread()
