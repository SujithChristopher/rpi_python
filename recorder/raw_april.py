import time
from picamera2 import Picamera2
import numpy as np
from numba import njit
from cv2 import aruco

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

picam2 = Picamera2()
config = picam2.create_video_configuration(
    raw=picam2.sensor_modes[2], controls={"FrameRate": 190}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
prev_frame_time = 0

new_frame_time = 0


@njit
def convert_to_8bit(image):
    image = image.astype(np.float32) / 65535
    return (image * 255).astype(np.uint8)


while 1:
    raw = picam2.capture_array("raw").view(np.uint16)

    img = convert_to_8bit(raw)
    corners, ids, rejected_image_points = detector.detectMarkers(img)
    corners, ids, _, _ = detector.refineDetectedMarkers(
        img, board, corners, ids, rejected_image_points
    )
    # gray_image = aruco.drawDetectedMarkers(gray_image, corners, ids)

    # cv2.imshow('alsdkfj', img)
    # cv2.waitKey(1)

    # metadata = picam2.capture_metadata()

    # print(raw.shape)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
