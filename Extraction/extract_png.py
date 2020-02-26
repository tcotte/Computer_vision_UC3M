# import the necessary packages
import cv2
import argparse
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input video file")
ap.add_argument("-o", "--output", required=True,
                help="path to output png file")
ap.add_argument("-f", "--frame", type=int, required=False,
                help="frame to extract from the video")
args = vars(ap.parse_args())

# Variables initialisation
video = cv2.VideoCapture(args["video"])
frame_extract = args["frame"]
print(frame_extract)
frame_counter: int = 0

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = video.read()
    frame_counter += 1
    
    # cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if frame_counter == frame_extract:
        # cv2.imshow("Output", frame)
        video.release()
        time.sleep(.300)
        cv2.imwrite(args["output"], frame)
        break

video.release()
cv2.destroyAllWindows()
