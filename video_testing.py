# -*- coding: utf-8 -*-

"""
Created on Fri May  8 06:53:26 2020

@author: Robomy
"""




import cv2
from deeplearn import loadModel,load_image

#load the model
model = loadModel()
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2

while(True):

    ret, frame = cap.read()
    new_image = load_image(frame)
    pred = model.predict(new_image)
    
    if(pred[0][0] > pred[0][1]):
                text = 'class1'
    else:
                text = 'class2'
    frame = cv2.putText(frame, text, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    #show the image
    cv2.imshow('frame', frame)
    #wait each frame, press q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()