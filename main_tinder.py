# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:56:40 2021

@author: KuroAzai

worked wonders for me maybe could for you also, use at your own risk of acc deletion etc...
"""

import cssutils
import keyboard
import shutil
import time
import os
import requests
import logging
from machine_learning import predictor
import argparse
from bs4 import BeautifulSoup
import config as cfg
from browser import web_engine
cssutils.log.setLevel(logging.CRITICAL)


def auto_tinder(number):
    while number > 0:
        soup = BeautifulSoup(src.driver.page_source, "html.parser")
        divs = soup.find_all("div", class_="Bdrs(8px) Bgz(cv) Bgp(c) StretchedBox")
        # print(len(divs))
        url = ''
        if len(divs) >= 2 and url == '':
            selector = divs[1]
            # print(x, type(x))
            soup = BeautifulSoup(str(selector), "html.parser")
            div_style = soup.find('div')['style']
            style = cssutils.parseStyle(div_style)
            # data = [x for x in style]
            # print(data)
            url = style['background-image']
            url = url.replace('url(', '').replace(')', '')
            temp_name = 'current_profile.jpg'
            response = requests.get(url, stream=True)
            with open(temp_name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            val = predictor.model_predict(temp_name)
            try:
                if keyboard.is_pressed('right'):
                    # save to liked
                    print('Liked')

                    file_count = count_files(LIKES)
                    name = LIKES + '/liked_{}.jpg'.format(str(file_count + 1))
                    response = requests.get(url, stream=True)
                    with open(name, 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                elif keyboard.is_pressed('left'):
                    # save to Disliked
                    print('Disliked')
                    file_count = count_files(DISLIKES)
                    name = DISLIKES + '/disliked_{}.jpg'.format(str(file_count + 1))
                    response = requests.get(url, stream=True)
                    with open(name, 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
            except ValueError:
                print(ValueError)

            if args.validation == 1:
                continue

            if val == 0:
                # like
                btn = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div/div[4]/div/div[4]/button'
                res = src.check_exists_by_xpath(btn, True)

                if res:
                    pass
                else:
                    btn = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div/div[5]/div/div[4]/button/span/span'
                    res = src.check_exists_by_xpath(btn, True)
                    if res:
                        pass
                    else:
                        continue

                path, dirs, files = next(os.walk(ML_LIKES))
                file_count = len(files) + 1
                name = ML_LIKES + '/m_liked_{}.jpg'.format(str(file_count + 1))
                shutil.move(temp_name, name)
                time.sleep(3)
                number -= 1
            else:
                # dislike
                btn = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div/div[5]/div/div[2]/button'
                res = src.check_exists_by_xpath(btn, False)
                if res:
                    pass
                else:
                    continue
                path, dirs, files = next(os.walk(ML_DISLIKES))
                file_count = len(files) + 1
                name = ML_DISLIKES + '/m_disliked_{}.jpg'.format(str(file_count + 1))
                shutil.move(temp_name, name)
                # save image to dislikes
                time.sleep(3)


def count_files(APP_FOLDER):
    total_files = 0
    total_dir = 0
    for base, dirs, files in os.walk(APP_FOLDER):
        for directories in dirs:
            total_dir += 1
        for Files in files:
            total_files += 1
    return total_files


def data_collection():
    while True:
        soup = BeautifulSoup(src.driver.page_source, "html.parser")
        divs = soup.find_all("div", class_="Bdrs(8px) Bgz(cv) Bgp(c) StretchedBox")
        # print(len(divs))
        url = ''
        if len(divs) >= 2 and url == '':
            selector = divs[1]
            soup = BeautifulSoup(str(selector), "html.parser")
            div_style = soup.find('div')['style']
            style = cssutils.parseStyle(div_style)
            url = style['background-image']
            url = url.replace('url(', '').replace(')', '')
            # print('\n', url)
        # like or dislike
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('right'):
                # save image into likes
                file_count = count_files(ML_LIKES)
                name = ML_LIKES + '/liked_{}.jpg'.format(str(file_count + 1))
                response = requests.get(url, stream=True)
                with open(name, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
            elif keyboard.is_pressed('left'):
                # save to dislikes
                file_count = count_files(ML_DISLIKES)
                name = ML_DISLIKES + '/disliked_{}.jpg'.format(str(file_count + 1))
                response = requests.get(url, stream=True)
                with open(name, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
        except ValueError:
            print(ValueError)


if __name__ == "__main__":
    src = web_engine.WebEngine()
    # terrible but too lazy to change it
    LIKES = cfg.liked_images
    DISLIKES = cfg.disliked_images
    ML_LIKES = cfg.mliked_images
    ML_DISLIKES = cfg.mdisliked_images

    # arg-parser for mode selection
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='data', help='The mode we want to run either data for data collection or auto for automatic swiping')
    parser.add_argument('--validation', type=int, default=1, help='Weahter we want to validate the model or not 1 or 0 for yes or no')
    args = parser.parse_args()
    if args.mode == 'auto':
        auto_tinder(cfg.total_likes)
    elif args.mode == 'data':
        data_collection()
