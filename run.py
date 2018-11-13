import numpy as np
import argparse
import cv2

class POI:
	def __init__(self, x, y, w, h, weight):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.weight = weight

points_of_interest = []

def analyzeFrame(image):
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	count = 0

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])

			if CLASSES[idx] != "bus" and CLASSES[idx] != "car":
				continue

			count += 1
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			point = ((endX - startX) / 2, (endY - startY) / 2)

			# display the prediction
			label = "Emergency vehicle at {}".format(point)
			print("[INFO] {}".format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	print("There were {} emergency vehicles in this picture.".format(count))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--list", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

images = args["list"].split(",")

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

for image in images:
	ext = image.split(".")[1]

	if ext == "bmp":
		analyzeFrame(cv2.imread(image))
	else if ext == "mp4":
		vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
		success,image = vidcap.read()
		count = 0
		while success:
		  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
		  success,image = vidcap.read()
		  print('Read a new frame: ', success)
		  count += 1
