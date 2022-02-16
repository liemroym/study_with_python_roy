from ctypes import string_at
from sys import prefix
import pyautogui
import time
import re

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

# for i in range(20):
#     pyautogui.press('tab')
#     pyautogui.hotkey('ctrl', 'v')
#     pyautogui.press('right')
#     pyautogui.press('left')

# for i in range(6, 48):
#     pyautogui.press('backspace')
#     pyautogui.write(str(i))
#     pyautogui.press('down')

# for i in range(2, 13):
#     pyautogui.press('f2')
#     pyautogui.write('Daspro_A2_' + str(i))
#     pyautogui.press('enter')
#     pyautogui.click(338, 180)
#     time.sleep(0.1)

lyric = '''
Super Idol的笑容
Super Idol de xiaorong
都没你的甜
dou mei ni de tian
八月正午的阳光
ba yue zhengwu de yangguang
都没你耀眼
dou mei ni yaoyan

热爱 105 °C的你
re’ai 105 °C de ni
滴滴清纯的蒸馏水
di di qingchun de zhengliushui
你不知道你有多可爱
ni bu zhidao ni you duo ke’ai
跌倒后会傻笑着再站起来
diedao hou hui shaxiaozhe zai zhan qilai

你从来都不轻言失败
ni conglai dou bu qing yan shibai
对梦想的执着一直不曾更改
dui mengxiang de zhizhuo yizhi buceng genggai
很安心 当你对我说
hen anxin dang ni dui wo shuo
不怕有我在
bupa you wo zai
放着让我来
fangzhe rang wo lai
勇敢追自己的梦想
yonggan zhui ziji de mengxiang
那坚定的模样
na jianding de muyang
Super Idol的笑容
Super Idol de xiaorong

都没你的甜
dou mei ni de tian
八月正午的阳光
ba yue zhengwu de yangguang
都没你耀眼
dou mei ni yaoyan
热爱 105 °C的你
re’ai 105 °C de ni

滴滴清纯的蒸馏水
di di qingchun de zhengliushui
在这独一无二
zai zhe duyiwu’er
属于我的时代
shuyu wo de shidai
不怕失败来一场
bupa shibai lai yi chang

痛快的热爱
tongkuai de re’ai
热爱 105°C的你
re’ai 105°C de ni
滴滴清纯的蒸馏水
di di qingchun de zhengliushui
在这独一无二
zai zhe duyiwu’er

属于我的时代
shuyu wo de shidai
莫忘了初心常在
mo wangle chuxin chang zai
痛快去热爱
tongkuai qu re’ai
热爱 105°C的你
re’ai 105°C de ni

滴滴清纯的蒸馏水
di di qingchun de zhengliushui
喝一口又活力全开
he yikou you huoli quan kai
再次回到最佳状态
zaici hui dao zui jia zhuangtai
喝一口哟
he yikou yo

你不知道你有多可爱
ni bu zhidao ni you duo ke’ai
跌倒后会傻笑着再站起来
diedao hou hui shaxiaozhe zai zhan qilai
你从来都不轻言失败
ni conglai dou bu qing yan shibai
对梦想的执着一直不曾更改
dui mengxiang de zhizhuo yizhi buceng genggai

很安心 当你对我说
hen anxin dang ni dui wo shuo
不怕有我在
bupa you wo zai
放着让我来
fangzhe rang wo lai
勇敢追自己的梦想
yonggan zhui ziji de mengxiang

那坚定的模样
na jianding de muyang
Super Idol的笑容
Super Idol de xiaorong
都没你的甜
dou mei ni de tian
八月正午的阳光
ba yue zhengwu de yangguang

都没你耀眼
dou mei ni yaoyan
热爱 105°C的你
re’ai 105°C de ni
滴滴清纯的蒸馏水
di di qingchun de zhengliushui
在这独一无二
zai zhe duyiwu’er

属于我的时代
shuyu wo de shidai
不怕失败来一场
bupa shibai lai yi chang
痛快的热爱
tongkuai de re’ai
热爱 105°C的你
re’ai 105°C de ni

滴滴清纯的蒸馏水
di di qingchun de zhengliushui
在这独一无二
zai zhe duyiwu’er
属于我的时代
shuyu wo de shidai
莫忘了初心常在
mo wangle chuxin chang zai

痛快去热爱
tongkuai qu re’ai
热爱 105°C的你
re’ai 105°C de ni
滴滴清纯的蒸馏水
di di qingchun de zhengliushui
喝一口又活力全开
he yikou you huoli quan kai

再次回到最佳状态
zaici hui dao zui jia zhuangtai
喝一口又活力全开
he yikou you huoli quan kai
'''
# try:
#     for i in range(28):
#         pyautogui.write('@Sahalul mabar uy')
#         pyautogui.press('enter')
# except KeyboardInterrupt:
#     pass

# for i in range(305):
#     pyautogui.press('f2')
#     # pyautogui.keyDown('ctrl')
#     # pyautogui.press('left')
#     # pyautogui.keyUp('ctrl')
#     pyautogui.press('backspace', 2)
#     pyautogui.press('enter')

# from direct_key import PressKey, down, enter, a, d

# for i in range(1):
#     PressKey(down)
#     PressKey(enter)
#     PressKey(d)
#     PressKey(down)
#     PressKey(enter, presses=2)
#     PressKey(d)
#     PressKey(down, presses=4)
#     PressKey(enter)

for i in range(20):
    pyautogui.leftClick(916, 863)
    time.sleep(0.2)


    