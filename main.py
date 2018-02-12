import cv2
import glob
import os
import imutils

path = "./BSR_bsds500/BSR/BSDS500/data/images/train"

data_set_input = []

for filename in glob.glob(os.path.join(path, '*.jpg')):
    image = cv2.imread(filename,0)

    rows, cols = image.shape
    
    if ((rows, cols) != (481, 321)):
        image = imutils.rotate_bound(image, 90)

    print(image.shape)

    data_set_input.append(image)

print (len(data_set_input))
