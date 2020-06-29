from djitellopy import Tello
import cv2
import time
######################################################################
width = 320  # WIDTH OF THE IMAGE
height = 240  # HEIGHT OF THE IMAGE
startCounter =0   #  0 FOR FIGHT 1 FOR TESTING
######################################################################


#make experiment folder

# CONNECT TO TELLO
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0

print(me.get_battery())

me.streamoff()
me.streamon()

me.takeoff()

counter=0
while True:

    # GET THE IMGAE FROM TELLO
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    print('myFrame shape '+str(myFrame.shape))
    img = cv2.resize(myFrame, (width, height))


    if(counter%10 ==0):
    	cv2.imwrite('image_'+str(counter).zfill(6)+'.png',myFrame)
    counter=counter+1

    # DISPLAY IMAGE
    cv2.imshow("Myr sult", img)

    key = cv2.waitKey(1) #& 0xff

    if key == 27: # ESC
        break
    elif key == ord('w'):
        me.move_forward(30)
    elif key == ord('s'):
        me.move_back(30)
    elif key == ord('a'):
        me.move_left(30)
    elif key == ord('d'):
        me.move_right(30)
    elif key == ord('e'):
        me.rotate_clockwise(30)
    elif key == ord('q'):
        me.rotate_counter_clockwise(30)
    elif key == ord('r'):
        me.move_up(30)
    elif key == ord('f'):
        me.move_down(30)

me.land()
