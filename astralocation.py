#!/usr/bin/env python

from calibration import Calibration
import cv2
from image_processing import find_features
from image_processing import Feature_Matcher
from longexposure import LongExposure
from matplotlib import pyplot as plt
import numpy as np
# from PIL import Image
from sensor_msgs.msg import CompressedImage
import rospy

class astraLocator(object):
	def __init__(self):
		rospy.init_node('astraLocator', anonymous=False)
		# self.bridge = CvBridge()
		self.image_received = False

		# Connect image topic
		img_topic = "/camera/image_raw/compressed"
		self.image_sub = rospy.Subscriber(img_topic, CompressedImage, self.compressedCallback)

		# Allow up to one second to connection
		rospy.sleep(1)

		if self.image_received:
			photos = self.collectPhotos()
			longExp = self.process(photos)
			(x,y) = self.getCoords(longExp)
			print x,y
		else:
			print "No image received."

	def compressedCallback(self, ros_data):
		''' 
		Callback function of subscribed topic. 
		Taken from http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber 
		'''

		#### direct conversion to CV2 ####
		np_arr = np.fromstring(ros_data.data, np.uint8)
		image_np = cv2.imdecode(np_arr, 1)

		self.image_received = True
		self.image = image_np

	def collectPhotos(self):
		""" Collects a set number of photos of the stars above the Neato. """
		flag = 0
		photos = []

		while (len(photos) <= 10) and (flag <= 20):
			if self.image_received:
				# image = np.array(self.image)
				photos.append(self.image)
			else:
				flag = flag + 1
		return photos

	def process(self, images):
		""" Filters or otherwise processes the data to be a single long-distance image. """
		le = LongExposure()
		for image in images:
			image = le.lightFilter(image)

		cv2.imshow('photo', images[0])
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		longExp = le.longExposure(images)

		# cv2.imshow('photo', longExp)
		# 	cv2.waitKey(0)
		# 	cv2.destroyAllWindows()
		return longExp

	def getCoords(self, img):
		""" Finds the coordinates of the robot. """
		starmap = cv2.imread('fin.png')
		fm = Feature_Matcher()
		MIN_MATCH_COUNT = 4

		usrKp, usrFt, usrCp = find_features(img) # keypoints, features, center_points
		mapKp, mapFt, mapCp = find_features(starmap)

		matches = Feature_Matcher.knn_vec_mating(usrFt, mapFt)

		#MATCH FILTERING
		good = []
		for m,n in matches:
			if m.distance < 0.55*n.distance:
				good.append(m)

		# Draw first 10 matches.
		img3 = cv2.drawMatches(img,usrKp,starmap,mapKp,good, None, flags=2)

		plt.imshow(img3),plt.show()

		if len(good)>=MIN_MATCH_COUNT:
			#translating from kp indices into coordinates
			src_pts = np.float32([ usrKp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ mapKp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			M2, matchesMask = self.find_homography(src_pts, dst_pts)
			col_im, new_im = self.find_map_match(img, starmap, M2, (usrCp, mapKp))

			cv2.imshow('mask', matchesMask)
			cv2.waitKey(1)
			cv2.destroyAllWindows()

			# Something about finding the center of the matched subpic
			# move origin -> (x-468.5, y-373)
			# conversion ratio x: 186.5 pixels/ft
			# conversion ratio y: 156.16 pixels/ft
			# Convert to feet
			center = (0,0)

			return center

		else:
			print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
			matchesMask = None

		x,y = 0,0
		return x,y

if __name__ == '__main__':
	al = astraLocator()