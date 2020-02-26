# Import the required packages
import argparse
import cv2

# We first create the ArgumentParser object
parser = argparse.ArgumentParser()

# We add 'path_image' argument using add_argument() including a help. The type of this argument is string (by default)
parser.add_argument("-p", "--path_image", required=True, help="path to input image to be displayed")

# The information about program arguments is stored in 'parser'
# Then, it is used when the parser calls parse_args().
# ArgumentParser parses arguments through the parse_args() method:
args = parser.parse_args()

# We can now load the input image from disk:
image = cv2.imread(args.path_image)

# Show the loaded image:
cv2.imshow("loaded image", image)

# Wait until a key is pressed:
cv2.waitKey(0)

# Destroy all windows:
cv2.destroyAllWindows()