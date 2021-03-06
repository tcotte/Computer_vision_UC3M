# USAGE
# python crop_image.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/toronto.mp4

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required=True,
                help="path to .txt file containing class labels")
ap.add_argument("-v", "--video", required=True,
                help="path to input video file")
ap.add_argument("-s", "--show", type=int, default=1,
                help="whether or not to display frame to screen")
ap.add_argument("-l", "--colors", type=str,
                help="path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default=500,
                help="desired width (in pixels) of input image")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args["colors"]:
    COLORS = open(args["colors"]).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

# otherwise, we need to randomly generate RGB colors for each class
# label
else:
    # initialize a list of colors to represent each class label in
    # the mask (starting with 'black' for the background/unlabeled
    # regions)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
                               dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["model"])

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["video"])
writer = None

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    total = -1

frame_counter: int = 0
# loop over frames from the video file stream
while True and frame_counter < 25:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    frame_counter += 1
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    if frame_counter == 20:
        cv2.imwrite('images/image20.png', frame)
        print('png file saved')
    # construct a blob from the frame and perform a forward pass
    # using the segmentation model
    frame = imutils.resize(frame, width=args["width"])

    # 1/255 -> integrate the color in [0 ; 1] interval
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()

    # infer the total number of classes along with the spatial
    # dimensions of the mask image via the shape of the output array
    (numClasses, height, width) = output.shape[1:4]

    # our output class ID map will be num_classes x height x width in
    # size, so we take the argmax to find the class label with the
    # largest probability for each and every (x, y)-coordinate in the
    # image
    classMap = np.argmax(output[0], axis=0)

    # given the class ID map, we can map each of the class IDs to its
    # corresponding color
    mask = COLORS[classMap]

    # resize the mask such that its dimensions match the original size
    # of the input frame
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    # perform a weighted combination of the input frame with the mask
    # to form an output visualization
    output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")


    # some information on processing single frame
    if total > 0:
        elap = (end - start)
        print("[INFO] single frame {} took {:.4f} seconds".format(frame_counter, elap))
        # print("[INFO] estimated total time: {:.4f}".format(elap * total))

    # write the output frame to disk

    # check to see if we should display the output frame to our screen
    if args["show"] > 0:
        cv2.imshow("Frame", output)
        cv2.imshow("Initial frames", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
