#!/usr/bin/env python

'''
Copyright (c) 2016, Nadya Ampilogova
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Modifications written by Lydia Zuehsow, CompRobo Fall 2018
'''

from cv_bridge import CvBridge, CvBridgeError
import cv2
import glob
import numpy as np
import os
from PIL import Image
import rospy
from sensor_msgs.msg import Image
import signal
from std_msgs.msg import String
import sys
import time

class Calibration(object):
	def __init__(self):
		self.bridge = CvBridge()
		self.image_received = False

		# Connect image topic
		img_topic = "/camera/image_raw"
		self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

		# Allow up to one second to connection
		rospy.sleep(1)
		
	def findCorners(self):
		# termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((6*7,3), np.float32)
		objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.

		if self.image_received:
			image = np.array(self.image)
			# image = np.array(Image.open("test.jpg").convert('RGBA'))
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			return cv2.findChessboardCorners(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), (7,6), None)
		else:
			return False, None

	def rectify(self):
		objpoints.append(objp)

		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
		cv2.imshow('img',img)
		cv2.waitKey(1)
		cv2.destroyAllWindows()

	def callback(self, data):

		# Convert image to OpenCV format
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		self.image_received = True
		self.image = cv_image

	def undistort(self, img):
		h,  w = img.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
		dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
		x,y,w,h = roi
		dst = dst[y:y+h, x:x+w]
		return dst

	def calibrate(self):
		ret = False
		flag = 0
		maxflag = 10

		for num in range(0,10):
			while ret is not True and flag <= maxflag:
				k = cv2.waitKey(1) & 0xFF
				ret, corners = self.findCorners()
				# print ret, corners
				flag = flag + 1
			if ret is True:
				self.rectify()
				# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
				ret = False

if __name__ == '__main__':
	# Initialize
	rospy.init_node('take_photo', anonymous=False)
	cal = Calibration()
	cal.calibrate()