from time import sleep
from picamera import PiCamera
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

# Create the in-memory stream
with PiCamera() as camera:
    camera.iso = 100
    sleep(2)
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g
    
    print('monitoring...')
    snapshot = np.empty((480, 640, 3), dtype=np.uint8)
    background = None
    try:
        while True:
            sleep(0.5) 
            camera.capture(snapshot, format='bgr', use_video_port=True, resize=(640, 480))
            cutout = snapshot[275:370, :530]
            frame = cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
            
            if background is None:
                background = frame.copy().astype('float')
                
            # accumulate the weighted average between the current frame and previous frames, then compute
            # the difference between the current frame and running average
            rc = cv2.accumulateWeighted(frame, background, 0.5)
            frameDelta = cv2.absdiff(frame, cv2.convertScaleAbs(background))
            
            # threshold the delta image, dilate the thresholded image to fill in holes,
            # then find contours on thresholded image
            thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            # look for large contour areas
            motion = False
            for cc in cnts:
                if cv2.contourArea(cc) < 300:
                    continue
            
                # compute the bounding box for the contour, draw it on the frame, and update the text
                motion = True
                (xx, yy, ww, hh) = cv2.boundingRect(cc)
                rc = cv2.rectangle(cutout, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
                #snapshot[275:370, :530] = cutout
                
            if motion:
                print('motion detected...saving')
                rc = cv2.imwrite('motion.jpg', cutout)
                rc = plt.imshow(cutout)
                plt.show()
    
    except KeyboardInterrupt:
        pass

