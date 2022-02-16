from ctypes import string_at
from msilib.schema import Directory
from sys import prefix
import pyautogui
import time
import re
import direct_key
import keyboard

def killer():
    raise KeyboardInterrupt

keyboard.add_hotkey('ctrl+shift+c', killer)

from pyscreeze import _locateAll_opencv

time.sleep(3)

# try:
#     for i in range(10):
#         pyautogui.leftClick(209, 136)
#         pyautogui.moveTo(308, 204)
#         time.sleep(0.5)
#         pyautogui.leftClick(560, 209)

#         pyautogui.press('down', presses=2)
# except KeyboardInterrupt:
#     print("\n")
    
# for i in range(30):
#     pyautogui.hotkey('shift', 'down')
#     pyautogui.leftClick(772, 172)
#     pyautogui.press('down')

# uy = input()

# haha = re.findall('\d+', uy)
# 100
# time.sleep(2)


# for i in range(len(haha)):
#     pyautogui.write(haha[i])
#     pyautogui.press('down')
#     pyautogui.press('down')
    
#     time.sleep(0.)

# print(haha)


# from direct_key import KeyDown, KeyUp, PressKey, ctrl, x

# KeyDown(ctrl)
    
# for i in range(50):
#     PressKey(x)
#     time.sleep(0.01)

# KeyUp(ctrl)

# ###########################################################
# #         MOUNT & BLADE WARBAND PEASANT TRAINING          #
# ###########################################################
# # Start from "Start practice fight" part (Train once first)
# import direct_key
# import winsound

# try:
#     while True:
#         pyautogui.leftClick(893, 710)
#         direct_key.left_click()
#         time.sleep(2)
#         direct_key.PressKey(direct_key.escape)
#         pyautogui.leftClick(502, 780)
#         direct_key.left_click()
#         time.sleep(0.3)
#         pyautogui.leftClick(416, 498)
#         direct_key.left_click()
#         time.sleep(0.3)
#         pyautogui.leftClick(1135, 832)
#         direct_key.left_click()
#         time.sleep(0.3)
#         pyautogui.leftClick(1523, 973)
#         direct_key.left_click()
#         time.sleep(2)
#         pyautogui.leftClick(939, 660)
#         direct_key.left_click()
#         time.sleep(0.3)
#         pyautogui.leftClick(882, 928)
#         direct_key.left_click()
#         time.sleep(0.3)
#         pyautogui.leftClick(893, 710)
#         direct_key.left_click()
#         time.sleep(7)
#         winsound.Beep(frequency=2500, duration=500)
#         time.sleep(3)
# except KeyboardInterrupt:
#     print("HIIII")

# AUTO CAPTURE
while True:
    pyautogui.leftClick(396, 732)
    direct_key.left_click()
    time.sleep(0.001)
    pyautogui.leftClick(936, 850)
    direct_key.left_click()
    time.sleep(0.001)