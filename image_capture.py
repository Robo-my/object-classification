# -*- coding: utf-8 -*-
"""
Created on Fri May  8 04:57:00 2020

@author: Robomy
"""

import cv2
cap = cv2.VideoCapture(0)

i = 0 # image id for saving
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #wait each frame, press s to save
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('images/'+str(i)+'.jpg',frame)
        i+=1 #incremnt id to avoid overwriting

    #show the image
    cv2.imshow('frame', frame)
    #wait each frame, press q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()