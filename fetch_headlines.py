from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from constants import *

# Create new text file if it doesn't exist
def create_file():
    f = open("headlines.txt", "w")
    f.close()


# Setup webdriver
def setup_driver():
    options = Options()
    options.binary_location = BINARY_LOCATION
    driver = webdriver.Chrome(executable_path=PATH_TO_WEB_DRIVER, chrome_options=options)
    return driver

# Navigate to URL
def navigate_to_url(driver):
    driver.get(BASE_URL)

# Click on the "Load More" button
def click_load_more(driver):
    load_more_button = driver.find_element(By.CLASS_NAME, BUTTON)
    ActionChains(driver).click(load_more_button).perform()

# Load all headlines onto page
def load_all_headlines(driver, count=100):
    for i in range(count):
        try:
            click_load_more(driver)
            sleep(1)
        except:
            break
    print("Successfully loaded headlines")

# Write all headlines to file
def write_headlines(driver):
    f = open("headlines.txt", "a", encoding="utf-8")
    headlines = driver.find_elements(By.CLASS_NAME, HEADLINE_ID)
    for headline in headlines:
        f.write(headline.text + "\n")
    f.close()
    print("Successfully wrote headlines to file")

# Increment date by 1 year
def increment_date(multiplier=0):
    start_ts = BASE_DATE + timedelta(days=multiplier*365)
    start_date = datetime.strftime(start_ts, "%m/%d/%Y")
    end_date = datetime.strftime(start_ts + timedelta(days=365), "%m/%d/%Y")
    return start_date, end_date

# Buffer for loading webpage
def buffer(driver):
    sleep(1)
    click_load_more(driver)
    sleep(1)
    click_load_more(driver)
    sleep(1)

# Specify date range as filter
def filter_by_date(driver):
    date_range_buttons = driver.find_elements(By.CLASS_NAME, DATE_RANGE_BUTTON)
    date_range_button = [i for i in date_range_buttons if i.text == "Specify date range"][0]
    ActionChains(driver).click(date_range_button).perform()

# Input date range
def input_date_range(driver, start_date, end_date):
    date_inputs = driver.find_elements(By.CLASS_NAME, "sc-1a9gghc-2.kYXNgQ")
    start_date_input = date_inputs[0]
    end_date_input = date_inputs[1]

    start_date_input.send_keys(Keys.CONTROL + "a");
    start_date_input.send_keys(Keys.DELETE);
    sleep(1)
    start_date_input.send_keys(start_date)
    sleep(1)
    start_date_input.send_keys(Keys.ENTER)
    sleep(1)
    end_date_input.send_keys(Keys.CONTROL + "a");
    end_date_input.send_keys(Keys.DELETE);
    sleep(1)
    end_date_input.send_keys(end_date)
    sleep(1)
    end_date_input.send_keys(Keys.ENTER)
    sleep(1)

# Main function
def main():
    create_file()
    driver = setup_driver()
    navigate_to_url(driver)
    buffer(driver)
    # Load headlines for each year
    year_diff = END_DATE.year - BASE_DATE.year
    for mult in range(0, year_diff):
        print("Loading headlines for year " + str(BASE_DATE.year + mult))
        start_date, end_date = increment_date(mult)
        filter_by_date(driver)
        input_date_range(driver, start_date, end_date)
        load_all_headlines(driver)
        write_headlines(driver)

    driver.close()


main()
