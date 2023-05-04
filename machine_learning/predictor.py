# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:05:05 2021

@author: KuroAzai
"""

from keras.models import load_model
import cv2
import os
import numpy as np

def model_predict(jpg):
    image = cv2.imread(jpg)
    image = cv2.resize(image, (320, 320))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)

    y_pred = model.predict(img)
    y_pred = np.round(y_pred).astype(int)
    if y_pred[0][0] == 0:
        return 0
    elif y_pred[0][0] == 1:
        return 1

# check if the model exists
if os.path.isfile('tinder_model_main.h5'):
    print('Model found')
    model = load_model('tinder_model_main.h5')
else:
    print('Model not found')
