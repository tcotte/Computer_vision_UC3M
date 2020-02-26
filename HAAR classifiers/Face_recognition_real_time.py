# import libraries
import cv2
import face_recognition

# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

# Program Files (x86)\IntelSWTools\openvino\opencv\etc\haarcascades
eye_cascade = cv2.CascadeClassifier(
    'C:\\Program Files (x86)\\IntelSWTools\\openvino\\opencv\\etc\\haarcascades\\haarcascade_eye.xml')

# Initialize variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #  detects objects and returns them as a list of rectangles
    eyes = eye_cascade.detectMultiScale(frame)
    for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(image, start_point(bottom-left), end_point(top-right), color, thickness)
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
