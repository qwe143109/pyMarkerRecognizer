import cv2
import numpy as np
import ThreadControl as tc

THRESHOLD = 10

capture=cv2.VideoCapture(0)
print capture.isOpened()
if capture.isOpened():
    tc.capture = capture
    tc.start_thread()
    while True:
        tc.lock_frame_out.acquire()
        if(tc.frame_out.size>0):
            cv2.imshow("src", tc.frame_out)
        tc.lock_frame_out.release()
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
    capture.release()
