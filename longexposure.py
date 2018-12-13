#!/usr/bin/env python

import argparse
import cv2
import imutils
import numpy as np
import os
from PIL import Image
import rospkg

"""
Need to fix time delay roslaunch
<node pkg="StarFinder" name="longexposure" type="longexposure.py" output="screen" args="$(find StarFinder)/data">
</node>
"""

class LongExposure(object):
	def __init__(self):
		pass

	@staticmethod
	def filterImages(image, path):
		filepath = os.path.join(path, image)
		npImage = np.array(Image.open(filepath).convert('RGBA'))
		return LongExposure.lightFilter(npImage)

	@staticmethod
	def lightFilter(img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1] # Threshold for light spots
		return thresh

	@staticmethod
	def longExposure(images):
		mask = images[0]
		for image in images[1::]:
			mask = mask + image
		return mask

	@staticmethod
	def processRosbagData():
		images = []
		rospack = rospkg.RosPack()
		path = os.path.join(rospack.get_path('StarFinder'), "data")

		os.system("rm -rf " + str(path) + "*.png")
		os.system("mv ~/.ros/frame*.jpg " + str(path))

		for num in range(len(os.listdir(path))):
			length = len(str(num))
			imagename = "frame" + "0"*(4 - length) + str(num) + ".jpg"
			images.append(LongExposure.filterImages(imagename, path))
		# cv2.imshow('filtered', images[0])
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		print "Saved long exposure image!"
		cv2.imwrite(os.path.join(rospack.get_path('StarFinder'),'le_x-2y-3t0.jpg'),LongExposure.longExposure(images))
		return

if __name__ == '__main__':
	# Data folder is 'ceiling_map'
	le = LongExposure()
	le.processRosbagData()