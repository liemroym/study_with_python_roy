# Test file

import time
import cv2
import numpy as np
import pyautogui

time.sleep(3)

#img = pyautogui.screenshot(region=(832, 263, 357, 79))
#img.save(r'D:\\1. College Stuff\\0. Programming stuff\\Python\\auto_clicker\\hello.png')

img = cv2.imread('D:\\1. College Stuff\\0. Programming stuff\\Python\\auto_clicker\\hello.png')

cv2.imshow('Hello', img)

cv2.waitKey(2)