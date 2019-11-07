import cv2
import os
import matplotlib.pyplot as plt
from postprocess import postprocess
from preprocess import preprocess
gt=os.listdir(os.path.join('HW1_dataset','groundtruth'))
gt=[int(path[5:8]) for path in gt]

backSub=cv2.bgsegm.createBackgroundSubtractorMOG(history=260,backgroundRatio=0.4)
capture=cv2.VideoCapture(cv2.samples.findFileOrKeep(os.path.join(os.path.join('HW1_dataset','input'),'in%06d.png')))
if capture.isOpened()==0:
    print('unable to open file')
    exit(0)

count=1
while True:
    ret,frame=capture.read()
    if frame is None:
        break

    fgMask=backSub.apply(preprocess(frame))
    fgMask=postprocess(fgMask,25,20)

    cv2.imshow('frame',frame)
    cv2.imshow('fgMask',fgMask)
    plt.show()
    cv2.waitKey(10)
    #save result (comparing with ground_truth)
    if count in gt:
        cv2.imwrite(os.path.join('EvaluationV2',os.path.join('results','{:0>6d}.png'.format(count))),fgMask)

    count += 1