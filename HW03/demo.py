from __future__ import division, print_function, absolute_import
import cv2
import time
import numpy as np
import os
from timeit import time
import warnings
import sys
from PIL import Image

import tello
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from control import tello_control
warnings.filterwarnings('ignore')

if __name__=='__main__':
    # yolo
    yolo=YOLO()

    # get drone object
    drone=tello.Tello('',8889)

    # set parameters
    W,H=960,720
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    writeVideo_flag = False

    # SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # write to result.avi
    if writeVideo_flag:
        out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (W, H))
        frame_index = -1


    fps = 0
    while True:
        t1=time.time() # to calculate fps
        # get frame from drone
        frame=drone.read()
        if frame is None:
            continue

        # keyboard interrupt
        key=cv2.waitKey(1)
        if key==27:  # key 'esc'
            break
        elif key==108:  # key 'l'
            drone.land()
        elif key==116:  # key 't'
            drone.takeoff()
            cv2.waitKey(7000)
            drone.move_up(1.0)
        elif key==119:  # key 'w'
            drone.move_forward(0.6)
        elif key==115:  # key 's'
            drone.move_backward(0.6)
        elif key==97:  # key 'a'
            drone.rotate_ccw(10)
        elif key==100:  # key 'd'
            drone.rotate_cw(10)
        elif key==113:  # key 'q'
            drone.move_up(0.3)
        elif key==101:  # key 'e'
            drone.move_down(0.3)

        # yolo detection
        image=Image.fromarray(frame)  # RGB
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)  # BGR
        boxs = yolo.detect_image(image)  # boxs[i]:[x,y,width,height]
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

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
        master_flag=1
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),
                          color=(255, 255, 255), thickness=2)
            cv2.putText(frame, text=str(track.track_id), org=(int(bbox[0]), int(bbox[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            # track a person with the smallest track_id
            if master_flag:
                tello_control(drone,[bbox[0],bbox[1],bbox[2],bbox[3]])
                master_flag = 0

        # detection: blue bounding box
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])), color=(255, 0, 0),
                          thickness=2)

        cv2.imshow('f', frame)

        if writeVideo_flag:
            out.write(frame)

        fps=(fps+1/(time.time()-t1))/2
        print('fps: {}'.format(fps))

    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()