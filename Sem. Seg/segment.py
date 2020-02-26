# python segment.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --image images/example_04.png
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
from color import closest_colour, get_colour_name


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required=True,
                help="path to .txt file containing class labels")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
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

# otherwise, we need to randomly generate RGB colors for each class label
else:
    # initialize a list of colors to represent each class label in the mask (starting with 'black')
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
                               dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

color_name = []

for c in COLORS:
    requested_colour = c
    actual_name, closest_name = get_colour_name(requested_colour)
    if actual_name is None:
        actual_name = closest_name
    color_name.append(actual_name)

array = [CLASSES, color_name]
# for i in range(len(array[0])):
#     print('{} corresponds to {}'.format(array[0][i], array[1][i]))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTorch(args["model"])
# net = cv2.dnn.readNetFromCaffe("Model/shelhamer/deploy.prototxt", args["model"])
# net = cv2.dnn.readNetFromTorch("Model/PSPnet.pth")

# load the input image, resize it, and construct a blob from it
image = cv2.imread(args["image"])
if image is None:
    print('Input image not found')
image = imutils.resize(image, width=args["width"])
# ENet was trained on was 1024x512
# blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
#                              swapRB=True, crop=False)
# caffe model
blob = cv2.dnn.blobFromImage(image, 1/255, (1024, 512), 0,
                             swapRB=True, crop=False)

# perform a forward pass using the segmentation model
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()
# show the amount of time inference took
print("[INFO] inference took {:.4f} seconds".format(end - start))

# infer the total classes number along with the spatial dimensions of the mask image via the shape of the output array
(numClasses, height, width) = output.shape[1:4]

# our output class ID map will be num_classes x height x width in size, so we take the argmax to find the class label
# with the largest probability for each and every (x, y)-coordinate in the image
classMap = np.argmax(output[0], axis=0)
x = []
print(classMap.flatten())
for e in classMap.flatten():
    if e > 0:
        if e not in x:
            x.append(e)

for count in range(len(array[0])):
    if count in x:
        print('{} corresponds to {}'.format(array[0][count], array[1][count]))
# given the class ID map, we can map each of the class IDs to its corresponding color
mask = COLORS[classMap]

# resize the mask and class map such that its dimensions match the original size of the input image
mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                  interpolation=cv2.INTER_NEAREST)
classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

# perform a weighted combination of the input image with the mask to form an output visualization
output = ((0.4 * image) + (0.6 * mask)).astype("uint8")

# show the input and output images
cv2.imshow("Input", image)
cv2.imshow("Output", output)
path_image = args["image"].split('/')
cv2.imwrite("images/Output/" + path_image[-1], output)
cv2.waitKey(0)
