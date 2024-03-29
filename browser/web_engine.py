import time
import re
from selenium import webdriver
from config import firefox_profile
from bs4 import BeautifulSoup
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import FirefoxProfile
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys


class WebEngine:

    def __init__(self):
        self.url = 'https://tinder.com'
        # options = Options()
        # Run Driver in headerless mode
        # options.headless = False
        self.profile = FirefoxProfile(firefox_profile)
        self.browser = webdriver.Firefox(self.profile)

    def load_page(self):
        self.browser.get(self.url)
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

    def get_current_image(self, xpath):
        element = self.browser.find_element(by='xpath', value=xpath)
        element_source = element.get_attribute('outerHTML')
        soup = BeautifulSoup(element_source, 'html.parser')

        # Find the div element
        div = soup.find('div')

        # Extract the value of the "style" attribute
        style = div.get('style')

        regex = r'url\("(.+)"\)'

        # Use the regular expression to find the text between url(" and ")
        matches = re.search(regex, style)
        if matches:
            url = matches.group(1)
            return url
        else:
            print("URL not found")

    def get_page_source(self):
        return self.browser.driver.page_source

    def refresh_page(self):
        self.browser.driver.refresh()

    def close_browser(self):
        self.browser.driver.close()

    def keyboard_left_click(self):
        self.browser.find_element_by_tag_name('body').send_keys(Keys.ARROW_LEFT)

    def keyboard_right_click(self):
        self.browser.find_element_by_tag_name('body').send_keys(Keys.ARROW_RIGHT)


