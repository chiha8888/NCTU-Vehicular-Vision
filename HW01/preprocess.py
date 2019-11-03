import cv2
import os

mask=cv2.imread(os.path.join('HW1_dataset','ROI.bmp'),0)

def preprocess(frame):
    re = cv2.bitwise_and(frame, frame, mask=mask)
    return re