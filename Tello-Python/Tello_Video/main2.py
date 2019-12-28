import tello
import cv2

if __name__=='__main__':
    drone=tello.Tello('',8889)
    while True:
        # get frame
        frame=drone.read()

        # s key to take off
        if cv2.waitKey(1) & 0xFF == ord('s'):
            drone.takeoff()
		# q key to land 
		if cv2.waitKey(1) & 0xFF == ord('q'):
	    	drone.land()
			break

        # process by frame
        cv2.imshow('frame', frame)
        #cv2.imwrite(p,cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
	'''
        # control drone
        drone.rotate_cw(20)
        drone.rotate_ccw(20)
        drone.move_forward(0.3)
	'''
	
