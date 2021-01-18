import threading
import functools as ft
import pyautogui
import os 
import os 
import sys
from playsound import playsound

try: 
	
	# creating a folder named data 
	if not os.path.exists('idle'): 
		os.makedirs('idle') 

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 

def screenshot(currentframe = 0):
    ss = pyautogui.screenshot(region=(500,80, 448, 448))
    name = './idle/frame' + str(currentframe) + '.jpg'
    ss.save(name)
    threading.Timer(1, ft.partial(screenshot, currentframe +1)).start()
    print(str(currentframe))
    ttikstop= str(currentframe)
    if ttikstop == '59' :
        playsound('fire.wav')
        os._exit(0)

screenshot()