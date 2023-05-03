import os
import time
from selenium import webdriver
from config import firefox_profile
from bs4 import BeautifulSoup
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import FirefoxProfile
from selenium.common.exceptions import NoSuchElementException


class WebEngine:
    def __init__(self):
        self.url = 'https://tinder.com'
        # options = Options()
        self.profile = FirefoxProfile(firefox_profile)
        self.browser = webdriver.Firefox(self.profile)
        # Run Driver in headerless mode
        # options.headless = False
        # Set the driver with its options
        # browser = webdriver.Firefox(profile)
        # Get the results from our headless browser for our objects page
        # driver = browser
        # time.sleep(1.5)
        # src.browser = browser

    def load_page(self):
        self.driver.get(self.url)
        time.sleep(1.5)

    def check_exists_by_xpath(self, xpath, cat):
        try:
            self.browser.driver.find_element_by_xpath(xpath)
        except NoSuchElementException:
            print('Click Unsuccessful', cat)
            time.sleep(1)
            return False
        self.browser.driver.find_element_by_xpath(xpath).click()
        print("Click Success", cat)
        return True


