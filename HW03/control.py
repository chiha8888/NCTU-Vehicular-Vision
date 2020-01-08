import time

rotate=0

def tello_control(drone, bbox):
    midpoint_x = (bbox[0] + bbox[2]) / 2
    midpoint_y = (bbox[1] + bbox[3]) / 2
    width = abs(bbox[2] - bbox[0])
    height = abs(bbox[1] - bbox[3])

    # forward & back
    if (height < 500):
        drone.move_forward(1.0)
    elif (height > 600):
        drone.move_backward(1.0)

    # rotate
    global rotate
    if midpoint_x<384 or midpoint_x>=576:
        rotate=rotate%2
        print('rotate: '+str(rotate))
        if rotate==0:
            if (midpoint_x < 192):
                drone.rotate_ccw(30)
            elif (midpoint_x >= 192 and midpoint_x < 384):
                drone.rotate_ccw(15)
            elif (midpoint_x >= 576 and midpoint_x < 768):
                drone.rotate_cw(15)
            elif (midpoint_x >= 768 and midpoint_x < 960):
                drone.rotate_cw(30)
        rotate+=1