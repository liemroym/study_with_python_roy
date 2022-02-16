# Didn't work: the screenshoting part works well, but image_to_string (reading text from the screenshots) doesn't work well with dw8xl fonts;
# Ideas to make it work: train the tesseract AI with dw8xl font, use image compare instead (get a few sample screenshots of attributes that you don't want to sell, ex: ss of induction, jubilation, etc)

from PIL.Image import new
import pytesseract
import time
import cv2
import numpy as np
import pyautogui

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

time.sleep(3)

#img = pyautogui.screenshot(region=(832, 263, 357, 79))
img = pyautogui.screenshot(region=(0, 0, 1366, 768))
img.save(r'D:\\1. College Stuff\\0. Programming stuff\\Python\\auto_clicker\\hello.png')

#cvImg = cv2.imread(r'D:\\1. College Stuff\\0. Programming stuff\\Python\\auto_clicker\\hello.png')
#newImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
#ret, img = cv2.threshold(np.array(img), 125, 255, cv2.THRESH_BINARY)

print(pytesseract.image_to_string(img))