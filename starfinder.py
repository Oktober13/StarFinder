#!/usr/bin/env python

import argparse
import cv2
import imutils
import numpy as np
import os
from PIL import Image
# from imutils import contours

class Starfinder(object):
	def __init__(self, args):
		images = []
		path = os.path.join(os.getcwd(),str(list(args.items()[0])[1]))

		if len(os.listdir(path)) == 0:
			print "Hoi"
			os.system("mv ~/.ros/frame*.jpg " + str(path))

		for num in range(len(os.listdir(path))):
			length = len(str(num))
			imagename = "frame" + "0"*(4 - length) + str(num) + ".jpg"
			images.append(Starfinder.filterImages(imagename, path))
		# mask = Starfinder.longExposure(images[258::])
		cv2.imshow('filtered', images[0])
		# cv2.imshow('longExposure', mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	@staticmethod
	def filterImages(image, path):
		filepath = os.path.join(path, image)
		npImage = np.array(Image.open(filepath).convert('RGBA'))
		return Starfinder.lightFilter(npImage)

	@staticmethod
	def lightFilter(img):
		# image = cv2.imread(args["image"])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# blurred = cv2.GaussianBlur(gray, (11, 11), 0)
		thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1] # Threshold for light spots
		# thresh = cv2.erode(thresh, None, iterations=2) # Clean up contours
		# thresh = cv2.dilate(thresh, None, iterations=4)
		return thresh

	@staticmethod
	def longExposure(images):
		mask = images[0]
		for image in images[1::]:
			mask = mask + image
		return mask

	@staticmethod
	def imgToPointCloud():
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts = imutils.contours.sort_contours(cnts)[0]

		for (i, c) in enumerate(cnts):
			((cX, cY), radius) = cv2.minEnclosingCircle(c)
			# Some stuff converting to point cloud?

if __name__ == '__main__':
	# Data folder is 'ceiling_map'
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--data", required=True, help="path to the data folder")
	args = vars(ap.parse_args())
	print args

	sf = Starfinder(args)