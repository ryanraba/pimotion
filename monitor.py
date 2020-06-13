import time
from picamera import PiCamera
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector
import cv2
import imutils
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

gdfolder = '1crBqym6aiqByFbqcQn0pJkt2YRWDbVz7'

#######################################################
# GDrive authorization
#######################################################
creds = []
SCOPES = ['https://www.googleapis.com/auth/drive.file']
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

drive = build('drive', 'v3', credentials=creds)


######################################################
# motion file setup
######################################################
# grab list of files that already exist
results = drive.files().list(fields="files(id, name)", q="'1crBqym6aiqByFbqcQn0pJkt2YRWDbVz7' in parents").execute()['files']
gfiles = [results[ii]['id'] for ii in range(len(results)) if results[ii]['name'].startswith('motion_')]

# make new placeholder files if necessary
for ii in range(len(gfiles), 10):
    rc = cv2.imwrite('motion.jpg', (np.random.rand(200,300,3)*255).astype(int))
    media = MediaFileUpload('motion.jpg', mimetype='image/jpg')
    gfile = drive.files().create(body={'name': 'motion_%s.jpg'%str(ii), 'parents':[gdfolder]}, media_body=media, fields='id').execute()
    gfiles += [gfile]


fig, ax = plt.subplots()
def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    ax.add_patch(rect)


#####################################################
# grab camera hardware and monitor
#####################################################
print('monitoring...')
snapshot = np.empty((1024, 1280, 3), dtype=np.uint8)
background = None
box = (0, 1280, 0, 1024)
gg = 0
state = 0  # 0 - no motion, 1 - initial motion, >1 - subsequent motion
prev_center, prev_timestamp = np.array([0,0]), 0.0

with PiCamera() as camera:
    time.sleep(2)
    try:
        while True:
            time.sleep(0.5) 
            camera.capture(snapshot, format='bgr', use_video_port=True, resize=(1280, 1024))
            
            # initialize things
            if background is None:
                ax.imshow(snapshot)
                rs = RectangleSelector(ax, line_select_callback, drawtype='box', rectprops=dict(fill=False), 
                                       minspanx=5, minspany=5, spancoords='data', interactive=False)
                plt.show()
                box = [int(ii) for ii in rs.extents]
                print(box)
            
            cutout = snapshot[box[2]:box[3], box[0]:box[1]]
            frame = cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
            
            if background is None:
                background = frame.copy().astype('float')
                
            # compute the difference between the current frame and running average
            frameDelta = cv2.absdiff(frame, cv2.convertScaleAbs(background))
            
            # threshold the delta image, dilate the thresholded image to fill in holes,
            # then find contours on thresholded image
            thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            # look for large contour areas
            motion, upload = False, False
            for cc in cnts:
                if cv2.contourArea(cc) < 300:
                    continue

                # can't deal with multiple things moving at once
                if motion == True:
                    motion = False
                    break
                
                # compute the bounding box for the contour, draw it on the frame, and update the text
                motion = True
                (xx, yy, ww, hh) = cv2.boundingRect(cc)
                center = np.array([xx + ww//2, yy + hh//2])
                timestamp = time.time()
                rc = cv2.rectangle(cutout, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
                #snapshot[box[2]:box[3], box[0]:box[1]] = cutout
            
            # initial motion
            if motion and state == 0:
                state = 1
                prev_center = center
                prev_timestamp = timestamp
            
            # last frame also had motion
            elif motion and state == 1:
                distance = np.sqrt(np.sum( np.square(center - prev_center) ))
                timedelta = timestamp - prev_timestamp
                
                print('motion detected...distance of {:1.0f} in {:3.1f} seconds'.format(distance, timedelta))
                rc = cv2.imwrite('motion.jpg', cutout)
                media = MediaFileUpload('motion.jpg', mimetype='image/jpg')
                gfile = drive.files().update(fileId=gfiles[gg], media_body=media).execute()
                gg = (gg + 1) % 10
                #rc = plt.imshow(cutout)
                #plt.show()
            
            # no motion
            else:  # accumulate more background
                state = 0
                rc = cv2.accumulateWeighted(frame, background, 0.5)
    
    except KeyboardInterrupt:
        pass

