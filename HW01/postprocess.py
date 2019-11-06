import cv2
import numpy as np

def postprocess(frame,dilate_num,erode_num):
    #denoise
    re=cv2.medianBlur(frame,5)
    re=cv2.morphologyEx(re,cv2.MORPH_CLOSE,np.ones((5,5)))

    kernel=np.ones((3,3),'int')
    re=cv2.erode(re,kernel,iterations=1)
    re=cv2.dilate(re,kernel,iterations=dilate_num)
    re=cv2.erode(re, kernel, iterations=erode_num)

    #close
    #kernel=cv2.getStructuringElement(cv2.MORPH_OPEN,(5,5))
    #re=cv2.morphologyEx(re,cv2.MORPH_CLOSE,kernel)
    return re