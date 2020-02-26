# import libraries
import cv2
import face_recognition
import json


with open('config.json') as config_file:
    data = json.load(config_file)

video_capture = cv2.VideoCapture('toronto.mp4')
if not video_capture.isOpened():
    print("Error opening video stream or file")


out = cv2.VideoWriter('out_gray.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                      (int(video_capture.get(3)), int(video_capture.get(4))))

frame_counter: int = 0

while video_capture.isOpened():
    # Capture frame by frame
    ret, frame = video_capture.read()

    if ret and frame_counter < 50:
        frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Display the resulting image
        cv2.imshow('Video', gray)
        # Recover video
        out.write(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# Release handle to the webcam
video_capture.release()
out.release()
cv2.destroyAllWindows()
