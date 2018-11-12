import numpy as np
import cv2
import matplotlib.pyplot as plt

class POI:
	def __init__(self, x, y, w, h, weight):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.weight = weight

points_of_interest = []

"""
def auto_canny(image):
	sigma = 0.33
	v = np.median(image)
 
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	return edged
"""


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
	cv2.imshow('edges', target_edges)
	for scale in np.linspace(0.08, 1.0, 30)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
		newX, newY = template.shape[1] * scale, template.shape[0] * scale
		scaled_template = cv2.resize(gray_template, (int(newX), int(newY)))
		template_edges = cv2.Canny(scaled_template, 30, 200)
		
		if target.shape[0] > newX and target.shape[1] > newY:
	
			result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCOEFF_NORMED)
			#threshold variable for matches
			threshold = 0.3
			loc = np.where(result >= threshold)
			for pt in zip(*loc[::-1]):
				points_of_interest.append(POI(int(pt[0]), int(pt[1]), int(pt[0] + newX), int(pt[1] + newY), 1))



#Main script:

ambulance = cv2.imread('resources/images/1.jpg')
template = cv2.imread('resources/images/star-of-life.jpg')
ScaledTemplateMatching(ambulance, template)
for point in points_of_interest:
	cv2.rectangle(ambulance, (point.x, point.y), (point.w, point.h), (0,255,255), 2)
	print(point)

cv2.imshow('final', ambulance)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

	


