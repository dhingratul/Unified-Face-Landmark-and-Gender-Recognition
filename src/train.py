#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:40:47 2017

@author: dhingratul
"""
import os
import pandas as pd
import utils
import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.datasets import fetch_lfw_people


mdir = '/media/dhingratul/Storage-unix/Dataset/LFW/'
lfw = 'lfw-deepfunneled/'
ann1 = 'annotations/male_names.txt'
ann2 = 'annotations/female_names.txt'
male = pd.read_table(mdir + ann1, header=None).as_matrix().tolist()
male_set = set([item for sublist in male for item in sublist])
female = pd.read_table(mdir + ann2, header=None).as_matrix().tolist()
female_set = set([item for sublist in female for item in sublist])
files = os.listdir(mdir+lfw)
# Model Selection
model = 1  # PCA
genData = False
if model == 0:
    # Deep Learning based
    if genData is True:
        X, y = utils.generateData(mdir, lfw, files, male_set, female_set)
        # Pickle
        Y = np.array(y)
        pickle.dump((X, Y), open("../model/data.p", "wb"))
        # SVM
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.1, random_state=0)
        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        sc = clf.score(X_test, y_test)
        print("Score for model {} is {}".format(model, sc))
        joblib.dump(clf, '../model/svm_model.pkl')
    else:
        # Read Pickle
        X, Y = pickle.load(open("../model/data.p", "rb"))
        # SVM
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.1, random_state=0)
        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        sc = clf.score(X_test, y_test)
        print("Score for model {} is {}".format(model, sc))
        joblib.dump(clf, '../model/svm_model.pkl')

elif model == 1:
    if genData is True:
        lfw_people = fetch_lfw_people(
                min_faces_per_person=1, funneled=True, resize=0.5)
        # Reshaped to (62 X 47)
        n_samples, h, w = lfw_people.images.shape
        X = lfw_people.data
        n_features = X.shape[1]
        names = lfw_people.target_names
        target = lfw_people.target
        Y = []
        MALE = utils.maleSet(male)
        for i in target:
            n = str(names[i])
            if n in MALE:
                Y.append(1)
            else:
                Y.append(0)
        pickle.dump((X, Y), open("../mode/data2.p", "wb"))
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.1, random_state=0)
        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        sc = clf.score(X_test, y_test)
        print("Score for model {} is {}".format(model, sc))
        joblib.dump(clf, 'svm_model2.pkl')
    else:
        X, Y = pickle.load(open("../model/data2.p", "rb"))
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.1, random_state=0)
        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        sc = clf.score(X_test, y_test)
        print("Score for model {} is {}".format(model, sc))
        joblib.dump(clf, '../model/svm_model2.pkl')
