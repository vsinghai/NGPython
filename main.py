import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import imutils

class POI:
	def __init__(self, x, y, w, h, weight):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.weight = weight

points_of_interest = []

# Takes in two images and applies scale-independent
# template matching to find template in target by scaling down
# template image multiple times and using edge detection to match.
def ScaledTemplateMatching(target, template):
	#edge detection on both images:
	gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
	gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	blurred_target = cv2.GaussianBlur(gray_target, (3, 3), 0)
	blurred_template = cv2.GaussianBlur(gray_template, (3, 3), 0)
	target_edges = cv2.Canny(blurred_target, 30, 200)

	for scale in np.linspace(0.08, 1.0, 30)[::-1]:
		# resize the image according to the scale
		newX, newY = template.shape[1] * scale, template.shape[0] * scale
		scaled_template = cv2.resize(gray_template, (int(newX), int(newY)))

		for angle in np.arange(0, 60, 10):
			rotated = imutils.rotate(scaled_template, angle)
			template_edges = cv2.Canny(rotated, 30, 200)
		
			if target.shape[0] > newX and target.shape[1] > newY:
	
				result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCOEFF_NORMED)
				#threshold variable for matches
				threshold = 0.3
				loc = np.where(result >= threshold)


				for pt in zip(*loc[::-1]):
					points_of_interest.append(POI(int(pt[0]), int(pt[1]), int(pt[0] + newX), int(pt[1] + newY), 1))



#variables used in text detecion
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
net = cv2.dnn.readNet('resources/east_text_detection.pb')

# uses the EAST text detector to detect any text in scene
# adds all detected text as a POI
def TextDetection(target, min_confidence):
	width = target.shape[1]
	height = target.shape[0]
	new_width = 320
	new_height = 320
	resized_target = cv2.resize(target, (new_width, new_height))
	blob = cv2.dnn.blobFromImage(resized_target, 1.0, (new_width, new_height), 
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates tha
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		
		for x in range(0, numCols):

			if scoresData[x] < min_confidence:
				continue
	
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
		
		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		width_ratio = width / float(new_width)
		height_ratio = height / float(new_height)

		startX = int(startX * width_ratio)
		startY = int(startY * height_ratio)
		endX = int(endX * width_ratio)
		endY = int(endY * height_ratio)
	
		points_of_interest.append(POI(startX, startY, endX, endY, 1))



#Main script:
print("started main")

target = cv2.imread('resources/images/air1.jpg')
template = cv2.imread('resources/images/filled-star-of-life.jpg')

#text_img = cv2.imread('resources/images/ambulance-back.jpg')
#TextDetection(text_img, 0.7)

ScaledTemplateMatching(target, template)

for point in points_of_interest:
	cv2.rectangle(target, (point.x, point.y), (point.w, point.h), (0,255,255), 2)

cv2.imshow("image", target)
cv2.waitKey(0)

#cv2.imshow('final', ambulance)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#ret, thresh = cv2.threshold(imgray, 120, 255, 0)
#im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnt = contours[0]
#cv2.drawContours(im2, contours, -1, (0,255,0), 3)
#x,y,w,h = cv2.boundingRect(cnt)
#cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)

"""
thresh = auto_thresh(ambulance)
ret, thresh = cv2.threshold(imgray, 120, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
for contour in contours:
	area = cv2.contourArea(contour)
	if area > 0:
		cv2.drawContours(ambulance, contour, -1, (0,255,0), 3)
		x,y,w,h = cv2.boundingRect(contour)
		cv2.rectangle(ambulance,(x,y),(x+w,y+h),(255,0,0),2)
"""

"""
def auto_canny(image):
	sigma = 0.33
	v = np.median(image)
 
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	return edged
"""
