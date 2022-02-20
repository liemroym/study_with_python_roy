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

data = '''
id_karyawan (pk)	nama_karyawan	NIK	nomor_telepon	jenis_kelamin	alamat	pekerjaan	gaji	bonus
7115823493	Bahuwarna Narpati	3104021304980028	+62818219390	L	Jl. Dr Muwardi Raya 36, Jakarta	Masinis	6000000	0.5
5286280652	Cornelia Yulianti	3275015203950021	+6282730770	F	Jl. Prambanan 3 Blok H1 No. 23, Bekasi	Paramedis	5500000	0.2
7482277128	Darmana Iswahyudi	1204011703820007	+6282138033	L	Gg. Lada No. 972, Gunungsitoli	Masinis	8000000	0.3
9293112306	Eman Gunarto	3571022311770002	+6282273215	L	Jl. Hayam Wuruk No. 20, Kediri	Kondektur	8000000	NULL
7032437267	Eva Usada	3578104507990034	+62831731907	F	Ds. Sutarto No. 932, Surabaya	Masinis	7000000	0.2
8874571375	Ifa Prastuti	1701046804920009	+62821310274	F	Kpg. Sudiarto No. 806, Bengkulu	Kondektur	8500000	0.7
5282890603	Ivan Saefullah	3277023112000023	+62817694448	L	Jl. Komplek Gn. Rahayu II No. 5, Bandung	Masinis	7000000	0.6
3626447933	Opung Tarihoran	3374092501890007	+62857488960	L	Jl. Karangroto No. 23, Semarang	Administrasi	6000000	NULL
8794615395	Soleh Permadi	1271160905970012	+62821850150	L	Jl. Mongonsidi No. 35, Medan	Teknisi Jalan Rel dan Jembatan	10000000	0.3
7877523720	Wulan Susanti	3274034702990053	+62821453275	F	Jl. Jend A. Yani No. 91, Cirebon	Teknisi Sarana Perkeretaapian	9000000	0.25
'''

data = data.split(sep='\n')

while (' ' in data):
    data.remove(' ')

for j in range(len(data)):
    data[j] = data[j].split('\t')

while ([''] in data):
    data.remove([''])

for line in data:
    for word in line:
        pyautogui.write(word)
        pyautogui.press('tab')

