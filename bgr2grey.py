import cv2 as cv


capture=cv.VideoCapture(0)
while True:
    isTrue, frame=capture.read()
    grey=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('Gray',grey)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()


