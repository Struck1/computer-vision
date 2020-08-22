from cv2 import cv2
import imutils
import copy
# shape no. rows (height) x no. columns (width) x no. channels (depth)
img = cv2.imread('../data_files/opencv_tutorial_load_image.jpg')

(h, w, d) = img.shape

print('width = {}, height={}, depth={}'.format(w, h, d))

cv2.imshow('Image', img)
cv2.waitKey(0)

# OpenCV stores images in BGR order rather than RGB
(B, G, R) = img[100, 50]

print('red = {}, green = {}, blue = {}'.format(R, G, B))

roi = img[60:160, 320:420]
# x = 320 y = 60, x = 420 y=160
cv2.imshow('Roı', roi)
cv2.waitKey(0)

# resize the image to 200x200px
resized = cv2.resize(img, (200, 200))
cv2.imshow('Resized', resized)
cv2.waitKey(0)

resized_i = imutils.resize(img, width=300)
print(resized_i.shape)
cv2.imshow("Imutils Resize", resized_i)
cv2.waitKey(0)


rotated = imutils.rotate(img, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

rotated = imutils.rotate_bound(img, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# blurred
blurred = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

output = img.copy()
rectangle = cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255))
cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)
# Since we are using OpenCV’s functions rather than NumPy operations we can supply our coordinates in (x, y)
# order rather than (y, x) since we are not manipulating or accessing the NumPy array directly — OpenCV is taking care of that for us.


output = img.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)

# draw green text on the image

output = img.copy()
#LEFT TOP 0,0
cv2.putText(output, 'OpenCV', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
cv2.imshow("Text", output)
cv2.waitKey(0)
