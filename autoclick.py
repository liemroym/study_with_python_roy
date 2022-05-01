from ctypes import string_at
from msilib.schema import Directory
from sys import prefix
from urllib.request import ProxyBasicAuthHandler
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

# # AUTO CAPTURE
# while True:
#     pyautogui.leftClick(396, 732)
#     direct_key.left_click()
#     time.sleep(0.001)
#     pyautogui.leftClick(936, 850)
#     direct_key.left_click()
#     time.sleep(0.001)

# coords = [(249, 275), (195, 351), (167, 442), (187, 545), (258, 614), (724, 274), (767, 379), (794, 447), (788, 553), (716, 638)]

# for coord in coords:
#     new_coords = coords.copy()
#     new_coords.remove(coord)

#     for ncoord in new_coords:
#         pyautogui.click(319, 102)
#         time.sleep(0.1)
#         pyautogui.click(coord)
#         time.sleep(0.1)
#         pyautogui.click(ncoord)
#         time.sleep(0.1)

#Point(x=249, y=275)
#int(x=195, y=351)  
#(x=167, y=442)
#(x=187, y=545
# (x=258, y=614
# x=767, y=379
# =794, y=447  
# 788, y=553
# 716, y=638
# 715, y=641
# Point(x=319, y=102)#

# #############################################
# #   AUTO ADD DATA INTO DATABASE WORKBENCH   #
# #############################################
# data = '''tortillahotel	railshon@zipet.site	~pvLMV93K1*M->BFiqA[tFm3#NOjA|?&-CtCEo9Z_UThp.@I\V[q^&uZYiyh
# lastoctopus	tgarjioni@yaachea.com	49vD#?X?&~+vN2d9soO3Z\8;@ZiZ5h-WV+BHc+RU=Dd8R9jBDqdd)bu_u}1U
# juvenilecowardice	kpwalls@guitarsxltd.com	UO\jQy%k:&J""gjWjdh'OkZ?[Z4-@R?cU5M=RpC0ogehd&dXAPOpvEVOBA-[
# icecreamcapricious	chicoos@osmye.com	Z*Sc&IQ{"3B9ON]jpk`ou:SLqth[J9jnhc<xI,<$G=B>F@;&V>V`&:Z&7(ma
# tumbletrail	lgspotlg@osmye.com	Clu]5=:r[}m-_)2g0aICe@/BFR=$g5VDx,O=5=i(,gjMzs|*#(0tq{S+LUpM
# '''

# data = data.replace('\n', ' ')
# data = data.replace('\t', ' ')
# data = data.split(' ')  

# for i, d in enumerate(data):
#     direct_key.write(d)
#     direct_key.PressKey(direct_key.keys['\t'])    

# for i in range(10000):
#     print(4, end=" ")


points = [(265, 251),
    (235, 365),
    (224, 497),
    (643, 268),
    (665, 373),
    (666, 481),
    (335, 667),
    (442, 664),
    (568, 640)]

for point in points:
    for p in points:
        if point != p:
            pyautogui.click(311, 107)
            pyautogui.click(point[0], point[1])
            pyautogui.click(p[0], p[1])
            
