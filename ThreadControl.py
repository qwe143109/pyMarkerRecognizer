import time
import thread
import numpy as np
import cv2

from MarketRecognizer import Marker, MarkerRecognizer


# global var
capture = 0
frame = np.array([])
frame_buffer = np.array([])
frame_out = np.array([])

# global class
m_recognizer = MarkerRecognizer()


# thread control
thread_run = False
lock_frame = thread.allocate_lock()
lock_frame_out = thread.allocate_lock()

# const
__FRAME_SIZE = (320, 240)


def start_thread():
    global thread_run
    thread_run = True
    thread.start_new_thread(thread_loadImage,())
    thread.start_new_thread(thread_CaptureProcess,())


def thread_loadImage():
    global frame
    print 'thread_loadImage'
    while thread_run:
        _, frm = capture.read()
        lock_frame.acquire()
        frame = cv2.resize(frm, __FRAME_SIZE)
        lock_frame.release()
        time.sleep(0.033)

def thread_CaptureProcess():
    global frame, frame_out
    print 'thread_CaptureProcess'
    while thread_run:
        lock_frame.acquire()
        n = frame.size
        lock_frame.release()
        if n>0:
            lock_frame.acquire()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_buffer = frame.copy()
            lock_frame.release()
            m_recognizer.update(img_gray, 20, 10)
            lock_frame_out.acquire()
            frame_out = frame_buffer
            m_recognizer.drawToImage(frame_out, 20, 10)
            cv2.line(frame_out, (0, 119), (319, 119), (0,0,255), 1)
            cv2.line(frame_out, (159, 0), (159, 239), (0,0,255) ,1)
            lock_frame_out.release()
            time.sleep(0.033)
        
