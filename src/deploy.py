#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:02:40 2017

@author: dhingratul
"""

from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
import utils
import scipy
from datetime import datetime


startTime = datetime.now()
clf1 = joblib.load('../model/svm_model.pkl')  # DL Model
clf2 = joblib.load('../model/svm_model2.pkl')  # PCA Model
model = '../model/shape_predictor_68_face_landmarks.dat'
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to input image",
                default="../data/0_1.jpg")
ap.add_argument("-m", "--model", required=False,
                help="model name: 'DL' for DL, 'PCA' for PCA", default='PCA')
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
raw_image = scipy.misc.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Male/ Female Prediction
    if args["model"] == 'DL':
        features = utils.featureExtract(image)
        features = np.reshape(features, (1, -1))
        out = clf1.predict(features)
    else:
        features = np.reshape(np.resize(image, (62, 47)), (1, -1))
        out = clf2.predict(features)
    if out == 0:
        text = 'Female'
    else:
        text = 'Male'
    # show the face number
    cv2.putText(image, "         {}".format(text), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    # print(text)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 255, 100), -1)

# show the output image with the face detections + facial landmarks
print('Time:', datetime.now() - startTime)
cv2.imshow("Output", image)
cv2.waitKey(0)
