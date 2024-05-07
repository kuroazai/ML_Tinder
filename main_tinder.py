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
        url = ''
        if len(divs) >= 2 and url == '':
            n = int(len(divs) / 2)
            selector = divs[n]
            soup = BeautifulSoup(str(selector), "html.parser")
            div_style = soup.find('div')['style']
            style = cssutils.parseStyle(div_style)
            url = style['background-image']
            url = url.replace('url(', '').replace(')', '')

            temp_name = 'current_profile.jpg'
            response = requests.get(url, stream=True)

            with open(temp_name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)

            val = predictor.model_predict(temp_name)

            if args.validation == 1:
                print('Validation mode')
                continue

            if val == 0:
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


def data_collection(seconds):
    print(f"please wait for {seconds} seconds")
    time.sleep(seconds)
    print("Starting data collection")
    while True:
        soup = BeautifulSoup(src.browser.page_source, "html.parser")
        divs = soup.find_all("div", class_="Bdrs(8px) Bgz(cv) Bgp(c) StretchedBox")
        url = ''
        if len(divs) >= 2 and url == '':
            n = int(len(divs) / 2)
            selector = divs[n]
            soup = BeautifulSoup(str(selector), "html.parser")
            div_style = soup.find('div')['style']
            style = cssutils.parseStyle(div_style)
            url = style['background-image']
            url = url.replace('url(', '').replace(')', '')

        if url == '' or url is None:
            continue

        if keyboard.is_pressed('right'):
            print('Liked')
            file_count = count_files(LIKES)
            name = LIKES + '/liked_{}.jpg'.format(str(file_count + 1))
            print(name)
            print(url)
            response = requests.get(url, stream=True)
            with open(name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
        if keyboard.is_pressed('left'):
            print('Disliked')
            file_count = count_files(DISLIKES)
            name = DISLIKES + '/disliked_{}.jpg'.format(str(file_count + 1))
            print(name)
            print(url)
            response = requests.get(url, stream=True)
            with open(name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)


def keyboard_test():
    print('Press a key')
    while True:
        event = keyboard.read_event()
        if event.event_type == 'down':
            key_value = event.name

            if key_value == 'left':
                print('Left arrow key pressed')
            elif key_value == 'right':
                print('Right arrow key pressed')


def check_folders():
    if not os.path.exists(LIKES):
        os.makedirs(LIKES)
    if not os.path.exists(DISLIKES):
        os.makedirs(DISLIKES)
    if not os.path.exists(ML_LIKES):
        os.makedirs(ML_LIKES)
    if not os.path.exists(ML_DISLIKES):
        os.makedirs(ML_DISLIKES)


def main(test=False):
    if test:
        keyboard_test()
        return
    check_folders()
    src.load_page()
    if args.mode == 'auto':
        print("running auto tinder")
        auto_tinder(cfg.total_likes)
    elif args.mode == 'data':
        print("running data collection")
        data_collection(10)


if __name__ == "__main__":
    src = web_engine.WebEngine()

    LIKES = cfg.liked_images
    DISLIKES = cfg.disliked_images
    ML_LIKES = cfg.mliked_images
    ML_DISLIKES = cfg.mdisliked_images

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='data', help='The mode we want to run either data for data collection or auto for automatic swiping')
    parser.add_argument('--validation', type=int, default=1, help='Weahter we want to validate the model or not 1 or 0 for yes or no')
    args = parser.parse_args()
    main()
