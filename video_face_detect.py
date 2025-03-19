import cv2 as cv

capture=cv.VideoCapture(0)
while True:
    isTrue, frame=capture.read()
    cv.imshow('Video webcam',frame)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('Gray',gray)
    tmp = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces=tmp.detectMultiScale(gray,1.1,6)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow('Faces',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()
