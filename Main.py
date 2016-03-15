import cv2
import numpy as np
from MarketRecognizer import Marker, MarkerRecognizer

capture=cv2.VideoCapture(0) #("http://192.168.18.218/videostream.cgi?user=admin&pwd=123456&.mjpg")
print(capture.isOpened())
m_recognizer = MarkerRecognizer()
if ( capture.isOpened() ):
    while True:
        ret, frame = capture.read()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m_recognizer.update(img_gray, 40, 20)
        m_recognizer.drawToImage(frame, (255,255,0),2)
        cv2.line(frame, (0, 239), (639, 239), (0,0,255), 1)
        cv2.line(frame, (319, 0), (319, 479), (0,0,255) ,1)
        
        
        mkCenters = m_recognizer.getMarkersCenter()
        if len(mkCenters) > 0:
            ppCenter = mkCenters.popitem()
            cv2.circle(frame, tuple(ppCenter[1]), 4, (0,0,255), 2)
            print ppCenter[1]
        cv2.imshow("src", frame)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    capture.release()
    