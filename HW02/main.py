import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from depthEstimate.detect import depth_estimation
from yolov3.detect import detect

import cv2

def draw_boxes(img, bbs):
    for bb in bbs:
        cv2.rectangle(img,bb[0],bb[1],(0,255,0),1)
    #cv2.imshow('frame',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def render_depth(img, bbs):
    for bb in bbs:
        depth = depth_estimation(img, bb)
        cv2.putText(img, "{:.2F}".format(depth), bb[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    #cv2.imshow('frame', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

video_name = 'test.avi'

#img=cv2.imread('road.jpg')
#bound_boxes=detect(img)
#draw_boxes(img,bound_boxes)
#render_depth(img,bound_boxes)
vc = cv2.VideoCapture(video_name)

while True:
    ret, frame = vc.read()
    if frame is None:
        break
    bound_boxes=detect(frame)
    draw_boxes(frame,bound_boxes)
    render_depth(frame,bound_boxes)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)