import numpy as np
import tensorflow as tf
from yolov3.util import *
import cv2
#print(tf.__version__)

model = tf.keras.models.load_model('yolov3/yolov3.h5')

def image_reshape(image,shape=(416, 416)):
    image=cv2.resize(image,shape)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    return image

def detect(image):
    h,w,c=image.shape
    image=image_reshape(image)
    output = model.predict(image)
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    boxes = []
    for i in range(len(output)):
        boxes += decode_netout(output[i][0], anchors[i], 0.5, 416, 416)
    #coordinate transfer
    correct_yolo_boxes(boxes, h, w, 416, 416)
    do_nms(boxes, 0.5)
    bbs = get_boxes(boxes, 0.5)

    return bbs


