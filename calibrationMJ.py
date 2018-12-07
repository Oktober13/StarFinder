#!/usr/bin/env python
import cv2
import glob
import numpy as np
import os
from PIL import Image

class Calibration(object):
	def __init__(self):
		# termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((6*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.
		images = ["photo4.jpg"]
		# images = glob.glob('*.jpg')
		for fname in images:
			#image = np.array(Image.open("photo.jpg").convert('RGBA'))
			image = cv2.imread(fname)
			#print(image.shape)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#print(gray.shape)
			ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
			#print(corners)

			if ret is True:
				print ("Hoi")
				objpoints.append(objp)

				corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners2)
				#print(objpoints)
				#print(imgpoints)
				# Draw and display the corners
				img = cv2.drawChessboardCorners(image, (6,9), corners2,ret)
				#cv2.imshow('img',img)
				cv2.imwrite("testcal.png", img)
		#cv2.waitKey(1)
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,(image.shape[1],image.shape[0]), None, None)
		print(type(dist))
		np.save("calmat.np",mtx)
		np.save("caldist.np",dist)
		testim=cv2.imread("photo4.jpg")
		h,w = testim.shape[:2]
		newcameramtx,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
		#print(newcameramtx)
		dst = cv2.undistort(testim, mtx, dist, None, newcameramtx)
		#x,y,w,h = roi
		#dst = dst[y:y+h, x:x+w]
		cv2.imwrite('testim.png',dst)
		#apx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
		#dst2 = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
		#x,y,w,h = roi
		#dst = dst[y:y+h, x:x+w]
		#cv2.imwrite('calibresult2.png',dst)
		mean_error = 0
		tot_error = 0
		for i in xrange(len(objpoints)):
			imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
			error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			tot_error += error
		print ("total error: ", mean_error/len(objpoints))

		#cv2.destroyAllWindows()

if __name__ == '__main__':
	cal = Calibration()
