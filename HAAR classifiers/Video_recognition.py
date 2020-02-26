# import libraries
import cv2
import face_recognition
import json


with open('config.json') as config_file:
    data = json.load(config_file)

video_capture = cv2.VideoCapture('Video_new.mp4')
if not video_capture.isOpened():
    print("Error opening video stream or file")
# Initialize variables
face_locations = []

out = cv2.VideoWriter('out_full.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                      (data['frame_width'], data['frame_height']))

frame_counter: int = 0

while video_capture.isOpened():
    # Capture frame by frame
    ret, frame = video_capture.read()

    if ret:
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        frame_counter += 1
        # Display the results
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)
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
