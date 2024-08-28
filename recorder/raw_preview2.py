import time
from picamera2 import Picamera2
import numpy as np
import cv2
from numba import njit


picam2 = Picamera2()
config = picam2.create_video_configuration(raw=picam2.sensor_modes[0], controls={'FrameRate':190})
picam2.configure(config)
picam2.start()
time.sleep(2)
prev_frame_time = 0

new_frame_time = 0

@njit
def convert_to_8bit(image):
    image = image.astype(np.float32)/65535
    return (image*255).astype(np.uint8)


while 1:
    raw = picam2.capture_array("raw").view(np.uint16)
    # print(np.max(raw))

    img = convert_to_8bit(raw)

    cv2.imshow('alsdkfj', img)
    cv2.waitKey(1)



    # metadata = picam2.capture_metadata()

    # print(raw.shape)

    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    print(fps)