#!/usr/bin/env python

from calibration import Calibration
import cv2
from image_processing import find_features
from image_processing import Feature_Matcher
from longexposure import LongExposure
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
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

		self.image_received = True
		self.image = image_np

	def collectPhotos(self):
		""" Collects a set number of photos of the stars above the Neato. """
		flag = 0
		photos = []

		while (len(photos) <= 20) and (flag <= 20):
			if self.image_received:
				image = np.array(self.image)
				photos.append(image)
			else:
				flag = flag + 1
		return photos

	def process(self, images):
		""" Filters or otherwise processes the data to be a single long-distance image. """
		le = LongExposure()
		for image in images:
			image = le.lightFilter(image)

		longExp = le.longExposure(images)
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

		if len(good)>=MIN_MATCH_COUNT:
			#translating from kp indices into coordinates
			src_pts = np.float32([ usrKp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ mapKp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			M2, matchesMask = self.find_homography(src_pts, dst_pts)
			col_im, new_im = self.find_map_match(img, starmap, M2, (usrCp, mapKp))

			cv2.imshow('mask', matchesMask)
			cv2.waitKey(1)
			cv2.destroyAllWindows()

			center = (0,0)
			for i, j in matches: # Trying to average the xy pos of the matches to find the center 
				center = (max(matches[i][j]) + min(matches[i][j])) / 2.0
			return center

		else:
			print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
			matchesMask = None

		x,y = 0,0
		return x,y

if __name__ == '__main__':
	al = astraLocator()