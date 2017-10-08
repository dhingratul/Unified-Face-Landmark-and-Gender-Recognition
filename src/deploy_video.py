#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:48:06 2017

@author: dhingratul
"""

# import the necessary packages
from sklearn.externals import joblib
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np


clf = joblib.load('../model/svm_model2.pkl')  # PCA Model
model = '../model/shape_predictor_68_face_landmarks.dat'
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks = np.array([
                # -1 for 0 based indexing in python
                shape[34 - 1],     # Nose tip
                shape[9 - 1],     # Chin
                shape[37 - 1],     # Left eye left corner
                shape[46 - 1],     # Right eye right corne
                shape[49 - 1],     # Left Mouth corner
                shape[55 - 1]      # Right mouth corner
                ])
        features = np.reshape(np.resize(frame, (62, 47)), (1, -1))
        out = clf.predict(features)
        if out == 0:
            text = 'Female'
        elif out == 1:
            text = 'Male'
        else:
            text = 'Not Recognized'
        cv2.putText(frame, "  {}".format(text), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
