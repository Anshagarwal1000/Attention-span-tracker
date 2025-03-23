import cv2
import numpy as np
import dlib
from imutils import face_utils

cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=detector(gray)
    for face in faces:
        landmarks=predictor(gray,face)
        
        for i in range (68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)    
    
    cv2.imshow("Face landmark",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     
cap.release()
cv2.destroyAllWindows()        