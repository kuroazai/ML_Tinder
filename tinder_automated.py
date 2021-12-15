# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:05:05 2021

@author: KuroAzai
"""

from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np


def model_predict(jpg):
    image = load_img(jpg, target_size=(320, 320))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)

    y_pred = model.predict(img)
    y_pred = np.round(y_pred).astype(int)
    if y_pred[0][0] == 0:
        return 0
    elif y_pred[0][0] == 1:
        return 1


model = load_model('tinder_model_main.h5')
