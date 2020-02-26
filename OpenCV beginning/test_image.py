# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)

# allow the camera to warmup
time.sleep(0.1)

# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# display the image on screen and 
cv2.imshow("Image", image)

# save the image to disk:
cv2.imwrite("image_out.png", image)

# wait for a keypress           
cv2.waitKey(0)

# close the instance of the camera
camera.close()

