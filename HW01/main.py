import cv2
import os
import matplotlib.pyplot as plt
from HW01.postprocess import postprocess
from HW01.preprocess import preprocess

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
    fgMask=postprocess(fgMask)

    cv2.imshow('frame',frame)
    cv2.imshow('fgMask',fgMask)
    plt.show()
    cv2.waitKey(10)


    if os.path.isfile(os.path.join(os.path.join('HW1_dataset','groundtruth'),'gt{:0>6d}.png'.format(count))):
        plt.figure()
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(plt.imread(os.path.join(os.path.join('HW1_dataset','groundtruth'),'gt{:0>6d}.png'.format(count))))
        plt.subplot(1,2,2)
        fgMask=cv2.cvtColor(fgMask,cv2.COLOR_BGR2RGB)
        plt.imshow(fgMask,cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join('vs','{}.png'.format(count)))
        plt.close()
    count+=1