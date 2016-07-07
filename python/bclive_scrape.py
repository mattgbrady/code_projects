# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:24:31 2016

@author: ABerner
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import numpy as np

dlpath = "C:\\Users\\andrewb\\Documents\\GitHub\\mrd\\"

fp = webdriver.FirefoxProfile()
fp.set_preference('browser.download.folderList', 2)
fp.set_preference('browser.download.manager.showWhenStarting', False)
fp.set_preference('browser.download.dir', dlpath)
fp.set_preference('browser.helperApps.neverAsk.saveToDisk', ('application/vnd.ms-excel'))

driver = webdriver.Firefox()
driver.get("https://live.barcap.com")
element = driver.find_element_by_name("user").send_keys("mawilson1")
element = driver.find_element_by_name("password").send_keys("Wurts2015")
element = driver.find_element_by_id("submit").click()
time.sleep(0.2)
driver.find_elements_by_link_text("here")[1].click()
np.random.uniform(0.0,0.5)
driver.get("https://live.barcap.com/BC/barcaplive?menuCode=MENU_IDX_1061")

index_group = 'U.S. Aggregate'
index_subgroup = 'Corporate'
index_name = 'Financial Institutions'

driver.switch_to_frame(driver.find_elements_by_tag_name("iframe")[3])
driver.find_element_by_link_text('U.S. Aggregate').click()
driver.find_element_by_link_text('Corporate').click()
driver.find_element_by_link_text('Time Series').click()
#def download_bcData(index_group,index_subgroup,index_name):
#    driver.switch_to_frame(driver.find_elements_by_tag_name("iframe")[3])
#    driver.find_element_by_link_text(index_group).click()
#    driver.switch_to_frame(driver.find_elements_by_tag_name("iframe")[3])
#    driver.find_element_by_link_text(index_subgroup).click()
#    driver.switch_to_frame(driver.find_elements_by_tag_name("iframe")[3])
#    driver.find_element_by_link_text(index_name).click()
#    driver.find_element_by_link_text("Time Series").click()
#    return

#download_bcData(index_group,index_subgroup,index_name)    
    
