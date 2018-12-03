# this file takes the image from the camera, locates all the stars, finds their centers and does some math
import statistics as stat
import os, sys
from os import path
import cv2 as cv2
import numpy as np
import math
from matplotlib import pyplot as plt
filepath = "/home/mj/catkin_ws/src/StarFinder/longExposure.png"
image = cv2.imread(filepath)

def Find_Features(im):
    """This function takes a long exposure image of the star ceiling and finds the feature vectors asssociated with it. """
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # turning the image grayscale
    ret,thresh = cv2.threshold(imgray,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #drawing the bounding contours for stars

    center_points = []
    radi = []
    for i in contours:
        center,radius = cv2.minEnclosingCircle(i)
        center_points.append(center)
        radi.append(radi)
        #cv2.circle(im, (int(center[0]),int(center[1])),int(radius),(0,255,0),-1)
        #im[int(center[1]),int(center[0])] = [0,0,255]
    #cv2.imwrite("test_boxes.png", im)

    dist_dict = {}
    #finding the star histagrams
    for n in center_points:
        dists = []
        for x in center_points:
            dist = math.sqrt(((x[0]-n[0])**2)+((x[1]-n[1])**2))
            if dist != 0:
                dists.append(dist)
        dist_dict[n] = dists

    feature_dict = {}
    vec_lists = []
    for z in dist_dict:
        #feature_vec = num stars in... [50,100,200] px and the end is the average dist to the closest 10 stars
        feature_vec = [0,0,0,0,0,0,0,0,0]
        d = dist_dict[z]
        d.sort()
        feature_vec[0]= d[0]
        feature_vec[1]=d[1]
        feature_vec[2]=d[2]
        feature_vec[3]=d[3]
        feature_vec[4]=d[4]
        for r in d:
            if r <= 50:
                feature_vec[6]+=1
                #feature_vec[1]+=1
                #feature_vec[2]+=1
                #feature_vec[3]+=1
            elif r <= 100:
                feature_vec[7]+=1
                #feature_vec[2]+=1
                #feature_vec[3]+=1
            elif r <= 200:
                feature_vec[8]+=1
                #feature_vec[3]+=1
            #elif r <= 200:
            #    feature_vec[9]+=1
        for s in d[0:10]:
            feature_vec[5]+=s
        feature_vec[5] = feature_vec[5]/10
        vec_lists.append(np.asarray(feature_vec))
        feature_dict[z] = feature_vec

    #similarity testing. Using cosign distintances between feature vectors to make sure they are unique enough
    """
    sims = []
    used_pairs = []
    for v1 in vec_lists:
        v1_l = v1.tolist()
        used_pairs.append((v1_l,v1_l))
        for v2 in vec_lists:
            v2_l = v2.tolist()
            if (v2_l,v1_l) not in used_pairs:
                cos = np.dot(v1,v2)/ (np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))
                sims.append(cos)
                used_pairs.append((v1_l,v2_l))
    print(stat.mean(sims))
    print(stat.stdev(sims))
    print(stat.variance(sims))
    #print(vec_lists)
    """
#Main loop
Find_Features(image)
