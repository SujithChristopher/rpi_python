import numpy as np
import cv2
from picamera2 import Picamera2
import libcamera
import toml
import keyboard
import msgpack as mp
import msgpack_numpy as mpn
import os
import time

frame_size = (1200, 800)

picam2 = Picamera2()
WIDTH = frame_size[0]
HEIGHT = frame_size[1]
main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
_c = {
    "FrameRate": 120,
    # 'ExposureTime':500
}
config = picam2.create_video_configuration(
    main, controls=_c, transform=libcamera.Transform(vflip=1)
)
picam2.configure(config)
picam2.start()

_fish_params = toml.load(os.path.join(os.getcwd(), "undistort_best.toml"))
_fish_matrix = np.array(_fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
_fish_dist = np.array(_fish_params["calibration"]["dist_coeffs"])

# chessboard_size = (6,4)

# _base_pth = '/home/sujith/Documents/programs/data/calibration'
# _file_name = 'calib_mono_chess_1200_800.msgpack'
# _file_name = 'test.msgpack'
# _file = open(os.path.join(_base_pth, 'calib_mono_chessboard_ff',_file_name), 'wb')

START_SAVING = False

prev_frame_time = 0

new_frame_time = 0

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    _fish_matrix, _fish_dist, np.eye(3), _fish_matrix, (1200, 800), cv2.CV_16SC2
)


while 1:
    frame = picam2.capture_array()[:HEIGHT, :WIDTH]
    frame = cv2.flip(frame, 1)

    # frame = cv2.undistort(frame, _fish_matrix, _fish_dist)
    frame = cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    # print(frame.shape)
    # ret, corners = cv2.findChessboardCorners(frame, chessboard_size)
    # # print(ret)
    # if ret:
    #     corners = cv2.cornerSubPix(frame, corners, (5, 5), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    #     frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

    _small_img = cv2.resize(frame, (250, 250))

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    if START_SAVING:
        packed = mp.packb(corners, default=mpn.encode)
        _file.write(packed)

    if keyboard.is_pressed("s"):
        START_SAVING = True
        print("started recording")

    if keyboard.is_pressed("q"):
        START_SAVING = False
        print("stopped recording, ending camera")
        break

    cv2.imshow("asdf", _small_img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
# _file.close()
