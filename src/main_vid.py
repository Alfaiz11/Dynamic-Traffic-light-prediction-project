#Currently onlt working for video!

from PIL import Image
import cv2
import numpy as np
import requests


video_url = r"D:\Mohammad Alfaiz\Dynamic-Traffic-light-prediction-project\data\raw\cars_raw_30fps.mp4"

cap = cv2.VideoCapture(video_url) #This makes a capture object
if (cap.isOpened() == False):
    print("Cant find video")

while(cap.isOpened()):
    #Main function (through while loop)
    ret, frame = cap.read() 
    if ret == True:
        #This is our pipeline 
        #cv2.imshow('Frame', frame)  # Here the given frame object is considered an image
        frame = cv2.resize(frame,(450,250), fx = 1, fy = 1  , interpolation=cv2.INTER_CUBIC)
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame,(5,5),0)
        dilate_frame = cv2.dilate(blur_frame,np.ones((3,3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        closing = cv2.morphologyEx(dilate_frame,cv2.MORPH_CLOSE,kernel)
        #This is our classifier 
        car_cascade_src = r"D:\Mohammad Alfaiz\Dynamic-Traffic-light-prediction-project\data\raw\cars.xml"
        car_cascade = cv2.CascadeClassifier(car_cascade_src)
        cars = car_cascade.detectMultiScale(closing, 1.1, 1)
        #This is setup for bounding box
        cnt = 0
        for (x, y, w, h) in cars:
            cv2.rectangle(dilate_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cnt += 1

        # Print the total number of detected cars and buses
        print(cnt, " cars found")
        #print(np.ones((3,3)))
        cv2.imshow('Car Classification', dilate_frame)
        #frame_arr = np.array(frame)
        #
        #cv2.imshow('GrayScale video Conversion', gray_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 
    else:
        break 


cap.release()
cv2.destroyAllWindows()

