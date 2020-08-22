import numpy as np
from cv2 import cv2
import argparse
import time

weights = '../data_files/yolov3.weights'
cfg = '../data_files/yolov3.cfg'
classes_coco = '../data_files/coco.names'

# load input image
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')

args = vars(ap.parse_args())


# load the COCO class labels
classes = []

with open(classes_coco, 'r') as f:
    classes = f.read().splitlines()

print(len(classes))

# colors, each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(classes), 3),
                           dtype="uint8")


# load the pre-trained YOLO model
net = cv2.dnn.readNet(weights, cfg)


# input images and dimensions
image = cv2.imread(args['image'])
H, W, _ = image.shape
print(W, H, _)

# we need from output layers
output_layers = net.getLayerNames()
output_layers = [output_layers[i[0] -1]
                       for i in net.getUnconnectedOutLayers()]

print(output_layers)

# construct a blob from the input image
#[blobFromImage] creates 4-dimensional blob from image. Optionally resizes and crops image from center,
# subtract mean values, scales values by scalefactor, swap Blue and Red channels.
blob = cv2.dnn.blobFromImage(
    image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img_blob)

net.setInput(blob)
start = time.time()
layerOutputs = net.forward(output_layers)
end = time.time()

print(" Time  {:.6f} seconds".format(end - start))


boxes = []
confidences = []
class_ids = []

for outputs in layerOutputs:
    for detection in outputs:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # filter out weak predictions 
        if confidence > 0.5:
            # returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                        0.4)

if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
      
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
