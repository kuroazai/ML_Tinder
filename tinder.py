# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:56:40 2021

@author: KuroAzai

worked wonders for me maybe could for you also, use at your own risk of acc deletion etc...
"""

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import FirefoxProfile
import cssutils
import keyboard
import shutil
import time
import os
import requests
import logging
import tinder_automated
from selenium.common.exceptions import NoSuchElementException
import argparse
cssutils.log.setLevel(logging.CRITICAL)


class manager():

    def __init__(self):
        self.driver = ''
        self.browser = ''


def load():
    url = 'https://tinder.com'
    # options = Options()
    profile = FirefoxProfile(r'your firefox profile here')
    driver = webdriver.Firefox(profile)
    # Run Driver in headerless mode
    # options.headless = False
    # Set the driver with its options
    # browser = webdriver.Firefox(profile)
    # Get the results from our headless browser for our objects page
    driver.get(url)
    # driver = browser
    # time.sleep(1.5)
    # src.browser = browser
    src.driver = driver


def count_files(APP_FOLDER):
    totalFiles = 0
    totalDir = 0
    for base, dirs, files in os.walk(APP_FOLDER):
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
    return totalFiles


# output folders for liked/disliked images
likes = r''
dislikes = r''
# output folder for models predictions when running
mlikes = r''
mdislikes = r''

src = manager()
load()


def check_exists_by_xpath(xpath, cat):
    try:
        src.driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        print('Click Unsuccessful', cat)
        time.sleep(1)
        return False
    src.driver.find_element_by_xpath(xpath).click()
    print("Click Success", cat)
    return True


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
            # check if my type or not
            val = tinder_automated.model_predict(temp_name)
            # like or dislike
            try:  # used try so that if user pressed other than the given key error will not be shown
                if keyboard.is_pressed('right'):
                    # save to liked
                    print('Liked')

                    file_count = count_files(likes)
                    name = likes + '/liked_{}.jpg'.format(str(file_count + 1))
                    response = requests.get(url, stream=True)
                    with open(name, 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                elif keyboard.is_pressed('left'):
                    # save to Disliked
                    print('Disliked')
                    file_count = count_files(dislikes)
                    name = dislikes + '/disliked_{}.jpg'.format(str(file_count + 1))
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
                # save image to likes
                res = check_exists_by_xpath(btn, True)

                if res:
                    pass
                else:
                    btn = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div/div[5]/div/div[4]/button/span/span'
                    res = check_exists_by_xpath(btn, True)
                    if res:
                        pass
                    else:
                        continue

                path, dirs, files = next(os.walk(mlikes))
                file_count = len(files) + 1
                name = mlikes + '/m_liked_{}.jpg'.format(str(file_count + 1))
                shutil.move(temp_name, name)
                time.sleep(3)
                number -= 1
            else:
                # dislike
                btn = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div/div[5]/div/div[2]/button'
                res = check_exists_by_xpath(btn, False)
                if res:
                    pass
                else:
                    continue
                path, dirs, files = next(os.walk(mdislikes))
                file_count = len(files) + 1
                name = mdislikes + '/m_disliked_{}.jpg'.format(str(file_count + 1))
                shutil.move(temp_name, name)
                # save image to dislikes
                time.sleep(3)


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
                file_count = count_files(mlikes)
                name = likes + '/liked_{}.jpg'.format(str(file_count + 1))
                response = requests.get(url, stream=True)
                with open(name, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
            elif keyboard.is_pressed('left'):
                # save to dislikes
                file_count = count_files(mdislikes)
                name = dislikes + '/disliked_{}.jpg'.format(str(file_count + 1))
                response = requests.get(url, stream=True)
                with open(name, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
        except ValueError:
            print(ValueError)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("validation")
    args = parser.parse_args()
    if args.mode == 'auto':
        auto_tinder(1000)
    elif args.mode == 'data':
        data_collection()
