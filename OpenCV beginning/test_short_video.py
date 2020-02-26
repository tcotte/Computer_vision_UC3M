# import libraries
import cv2
import face_recognition
from picamera.array import PiRGBArray
from picamera import PiCamera
from subprocess import call
import time
from time import sleep

# Get reference of video
video = cv2.VideoCapture("/home/pi/Videos/Video_new.mp4")
# Check if camera opened successfully
if not video.isOpened():
    print("Error opening video stream or file")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_shortvid.avi', fourcc, 20.0, (640, 480), True)

# Initialize variables
face_locations = []
nb_frame = 1

# Capture frame-by-frame from the camera
success, frame = video.read()
count = 0

while (video.isOpened() and count <50):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
    out.write(frame)
    print (count)
    cv2.imshow('short_video', frame)
    count +=1


cv2.destroyAllWindows()
video.release()
out.release()

