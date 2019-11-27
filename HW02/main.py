import cv2
from yolov3.detect import detect
from depthEstimate.detect import depth_estimation

def draw_boxes(img, bbs):
    for bb in bbs:
        cv2.rectangle(img,bb[0],bb[1],(0,255,0),1)

def render_depth(img, bbs):
    for bb in bbs:
        depth = depth_estimation(img, bb)
        cv2.putText(img, "{:.2f}".format(depth), bb[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

video_name = 'test.avi'
vc = cv2.VideoCapture(video_name)

while True:
    ret, frame = vc.read()
    if frame is None:
        break
    k=cv2.waitKey(1) #'q' key to quit
    if k==113:
        break
    bound_boxes=detect(frame)
    draw_boxes(frame,bound_boxes)
    render_depth(frame,bound_boxes)
    cv2.imshow('frame', frame)


cv2.destroyAllWindows()