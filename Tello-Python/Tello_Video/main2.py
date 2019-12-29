import cv2
import time
import numpy as np
import tello

if __name__=='__main__':
    drone=tello.Tello('',8889)
    out=cv2.VideoWriter('output4.avi',cv2.VideoWriter_fourcc(*'XVID'),100,(960,720))
    fps=0
    while True:
        t1=time.time()
        # get frame
        frame=drone.read()
        if frame is None:
            continue

        key=cv2.waitKey(100)
        if key==27:  # key 'esc'
            drone.land()
            break
        elif key==108:  # key 'l'
            drone.land()
        elif key==116:  # key  't'
            drone.takeoff()
        elif key==119:  # key 'w'
            drone.move_forward(0.3)
        elif key==115:  # key 's'
            drone.move_backward(0.3)
        elif key==97:  # key 'a'
            drone.rotate_ccw(20)
        elif key==100:  # key 'd'
            drone.rotate_cw(20)
        elif key==113:  # key 'q'
            drone.move_up(1.3)
        elif key==101:  # key 'e'
            drone.move_down(1.3)

        # process frame
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imshow('f', frame)
        out.write(frame)
        fps=(fps+1/(time.time()-t1))/2
        print('fps: {}'.format(fps))

    out.release()
    cv2.destroyAllWindows()