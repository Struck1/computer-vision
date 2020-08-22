import argparse
import imutils
from cv2 import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')

print(ap.parse_args())
args = vars(ap.parse_args())
print(args)

image = cv2.imread(args['image'])
cv2.imshow("Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)


edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

# print(cnts[0])
# cnts = imutils.grab_contours(cnts)
# print(cnts[0])

outputs = image.copy()

for c in cnts[0]:
    cv2.drawContours(outputs, [c], -1, (240, 0, 159), 3)
    cv2.imshow("Contours", outputs)
    cv2.waitKey(0)


text = "found {} objects!".format(len(cnts[0]))
cv2.putText(outputs, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
cv2.imshow("Contours", outputs)
cv2.waitKey(0)


# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("bitw",output)
cv2.waitKey(0)