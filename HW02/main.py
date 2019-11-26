import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolov3.detect import detect
import cv2

def draw_boxes(img, bbs):
    for bb in bbs:
        cv2.rectangle(img,bb[0],bb[1],(0,255,0),1)
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img=cv2.imread('road.jpg')
bound_boxes=detect(img)
draw_boxes(img,bound_boxes)