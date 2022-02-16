# Don't know what is this, probably tried to use library to trigger keyboard  other than pyautogui and direct_key

import time
from pynput.keyboard import Key, Controller

keyboard = Controller()

time.sleep(6)

i = 0

while (i < 99):
    keyboard.press(Key.page_down)
    keyboard.release(Key.page_down)
    keyboard.press(Key.down)
    keyboard.release(Key.down)