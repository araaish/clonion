from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from constants import *

# Create new text file if it doesn't exist
def create_file():
    f = open("headlines.txt", "a")
    f.close()


# Setup webdriver
def setup_driver():
    options = Options()
    options.binary_location = BINARY_LOCATION
    driver = webdriver.Chrome(executable_path=PATH_TO_WEB_DRIVER, chrome_options=options)
    return driver

# Navigate to URL
def navigate_to_url(driver):
    driver.get(URL)

# Click on the "Load More" button
def click_load_more(driver):
    load_more_button = driver.find_element(By.CLASS_NAME, BUTTON)
    ActionChains(driver).click(load_more_button).perform()

# Load all headlines onto page
def load_all_headlines(driver):
    while True:
        try:
            click_load_more(driver)
            sleep(1)
        except:
            break

# Write all headlines to file
def write_headlines(driver):
    f = open("headlines.txt", "a")
    headlines = driver.find_elements(By.CLASS_NAME, HEADLINE_ID)
    for headline in headlines:
        f.write(headline.text + "\n")
    f.close()

# Main function
def main():
    create_file()
    driver = setup_driver()
    navigate_to_url(driver)
    load_all_headlines(driver)
    write_headlines(driver)
    driver.close()

main()