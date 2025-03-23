import cv2
import numpy as np
import dlib
from imutils import face_utils
from playsound import playsound
import threading
sound_playing = False 
def play_sound():
    global sound_playing
    if not sound_playing:
        sound_playing = True
        playsound("music.wav")  
        sound_playing = False
