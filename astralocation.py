#!/usr/bin/env python

from calibration import Calibration
from cv_bridge import CvBridge, CvBridgeError
import cv2
from image_processing import find_features, random_image_subsection
from image_processing import Feature_Matcher
from longexposure import LongExposure
from matplotlib import pyplot as plt
import numpy as np
import os
# from PIL import Image
from sensor_msgs.msg import Image
import rospy

class astraLocator(object):
	def __init__(self):
		rospy.init_node('astraLocator', anonymous=False)
		self.bridge = CvBridge()
		self.image_received = False

		# Connect image topic
		img_topic = "/camera/image_raw"
		self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

		# Allow up to one second to connection
		rospy.sleep(1)
		le = LongExposure()

		bagname = "x_0_y_-3_t_0"
		os.system("roslaunch imgExport.launch bag:=rosbags/" + bagname + ".bag")
		longExp = le.processRosbagData(bagname)
		(x,y) = self.getCoords(longExp)
		print x,y

		# if self.image_received:
		# 	longExp = le.processRosbagData(bagname)
		# 	# photos = self.collectPhotos()
		# 	# longExp = self.process(photos)
		# 	(x,y) = self.getCoords(longExp)
		# 	print x,y
		# else:
		# 	print "No image received."

	def callback(self, data):

		# Convert image to OpenCV format
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			cameraMatrix = np.array([498.4101369341172, 0, 321.1079043926777, 0, 497.5069069537356, 241.5754643292645, 0, 0, 1])
			distCoeffs = np.array([0.1344185941087997, -0.2322439415312437, -0.00230698609453406, -0.00261948989945836, 0])

			# dst = cv2.undistort(cv_image, cameraMatrix, distCoeffs)
			# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
		except CvBridgeError as e:
			print(e)

		self.image_received = True
		self.image = cv_image

	def collectPhotos(self):
		""" Collects a set number of photos of the stars above the Neato. """
		flag = 0
		photos = []

		while (len(photos) <= 10) and (flag <= 20):
			if self.image_received:
				image = self.image
				photos.append(image)
			else:
				flag = flag + 1
		return photos

	def process(self, images):
		""" Filters or otherwise processes the data to be a single long-distance image. """
		le = LongExposure()
		filtered = []
		for image in images:
			filtered.append(le.lightFilter(image))

		longExp = le.longExposure(filtered)
		# cv2.imshow('longExposure', longExp)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		return longExp

	def getCoords(self, img):
		""" Finds the coordinates of the robot. """
		le = LongExposure()
		starmap = le.lightFilter(cv2.imread('big_map.png'))

		img = cv2.dilate(img, np.ones((3,3), dtype=np.uint8))
		starmap = cv2.dilate(starmap, np.ones((3,3), dtype=np.uint8))

		# n = 512
		# m = 512

		# n_astar = 256
		# n_bstar = 64

		# # source map
		# starmap = np.zeros((n,m), np.uint8)
		# star_i = np.random.randint(n, size=n_astar)
		# star_j = np.random.randint(m, size=n_astar)
		# starmap[star_i, star_j] = 255

		# # bullshit image
		# img     = np.zeros((n,m), np.uint8)
		# img[256:, 256:] = np.copy(starmap[:256, :256])
		# star_i = np.random.randint(n, size=n_bstar)
		# star_j = np.random.randint(m, size=n_bstar)
		# img[star_i, star_j] = 255

		# # source map
		# star_i = np.random.randint(n, size=n_bstar)
		# star_j = np.random.randint(m, size=n_bstar)
		# starmap[star_i, star_j] = 255

		fm = Feature_Matcher()
		MIN_MATCH_COUNT = 4

		# section = random_image_subsection('big_map.png')

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
		print(len(good))
		plt.imshow(img3)
		plt.show()
		cv2.waitKey(0)

		if len(good)>=MIN_MATCH_COUNT:
			h,w = img.shape[0:2]
			#translating from kp indices into coordinates
			src_pts = np.float32([ usrKp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ mapKp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2) # the size of our image

			M2, matchesMask = fm.find_homography(src_pts, dst_pts) # matches the two images up.

			# since M2 is actually a rotation matrix and a translation vector packed together, we need to unpack them
			M2_m = np.array([(M2.tolist()[0][0:2]),(M2.tolist()[1][0:2])])
			M2_a = np.array([[(M2.tolist()[0][2])],[(M2.tolist()[1][2])]])
			dst = pts
			for p in dst:
				p[0] = np.transpose(M2_m.dot(np.asarray([[p[0][0]],[p[0][1]]]))+M2_a)
			#col_im, new_im = self.find_map_match(img, starmap, M2, (usrCp, mapKp))
			#dst contains the points for all corners of new image overlay
			new_c = ((dst[2][0][0]-dst[0][0][0])/2,(dst[2][0][1]-dst[0][0][1])/2)


			# cv2.imshow('mask',matchesMask)
			# cv2.waitKey(1)
			# cv2.destroyAllWindows()

			# Something about finding the center of the matched subpic
			# move origin -> (x-468.5, y-373)
			# conversion ratio x: 186.5 pixels/ft
			# conversion ratio y: 156.16 pixels/ft
			# Convert to feet
			center = ((new_c[0]-468.5)/186.5,(new_c[1]-468.5)/156.16 + 1.4)

			return center

		else:
			print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
			matchesMask = None

		x,y = 0,0
		return x,y

if __name__ == '__main__':
	al = astraLocator()