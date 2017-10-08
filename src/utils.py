#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:06:06 2017

@author: dhingratul
"""
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import scipy
import numpy as np


def imCrop(image):
    model = '../model/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model)
    # load the input image, resize it, and convert it to grayscale
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        out = image[x:(x+w), y:(y+h)]
    return out


def generateData(mdir, lfw, files, male_set, female_set):
    y = []
    x = files[0]
    im = os.listdir(mdir + lfw + x)
    filename = im[0]
    if filename in male_set:
        y.append(1)
    elif filename in female_set:
        y.append(0)
    im = scipy.misc.imread(mdir + lfw + x + '/' + filename)
    im_c = featureExtract(im)
    X = np.array(im_c)

    for i, x in enumerate(files[1:]):
        if i % 100 == 0:
            print(i)
        im = os.listdir(mdir + lfw + x)
        filename = im[0]
        im = scipy.misc.imread(mdir + lfw + x + '/' + filename)
        im_c = featureExtract(im)
        temp = np.array(im_c)
        if temp.size == 128 and filename in male_set:
            # print("Male", i+1)
            y.append(1)
            X = np.vstack((X, temp))
        elif temp.size == 128 and filename in female_set:
            # print("Female", i+1)
            y.append(0)
            X = np.vstack((X, temp))
    return X, y


def featureExtract(img):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1(
            '../model/dlib_face_recognition_resnet_model_v1.dat')
    dets = detector(img, 1)
    # Now process each face we found.
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return face_descriptor


def maleSet(male):
    ls = []
    for i in male:
        x = i[0].split('_')
        ls.append(" ".join(x[:-1]))
    return set(ls)
