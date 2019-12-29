#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture('output.avi')

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w,h = int(video_capture.get(3)),int(video_capture.get(4))
        out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    f=0
    while True:
        t1 = time.time()
        ret, frame = video_capture.read()
        if ret != True:
            break
        f+=1
        if f%100!=0:
            continue
        else:
            f=0

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image) #boxs[i]:[x,y,width,height]
        # print("box_num",len(boxs))
        features = encoder(frame,boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]


        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # track(predicted bounding of existing targets): white bounding box
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),color=(255,255,255), thickness=2)
            cv2.putText(frame, text=str(track.track_id),org=(int(bbox[0]), int(bbox[1])),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),thickness=2)

        # detection: red bounding box
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),color=(0,0,255), thickness=2)

        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps+(1./(time.time()-t1)))/2
        print("fps= %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
