# import libraries
import cv2

# Get reference of video
video = cv2.VideoCapture("10degrees_out.avi")
# Check if camera opened successfully
if not video.isOpened():
    print("Error opening video stream or file")

# Start and end of the output video according to this video
start: int = 0
end: int = 1

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('Output/out_shortvid_74.avi', fourcc, 20.0, (400, 266), True)

# Initialize variables
face_locations = []

# Capture frame-by-frame from the camera
success, frame = video.read()
count = 0
print(frame.shape)

while success:
    cv2.imshow('short_video', frame)
    count += 1
    if count > 74*25:
        print('[INFO] printing')
        out.write(frame)
        if count < 85*25:
            break


cv2.destroyAllWindows()
video.release()
out.release()
