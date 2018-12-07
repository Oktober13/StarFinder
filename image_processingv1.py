# this file takes the image from the camera, locates all the stars, finds their centers and does some math
import statistics as stat
import os, sys
from os import path
import cv2 as cv2
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from PIL import Image
filepath = "/home/mj/catkin_ws/src/StarFinder/longExposure.png"
image = cv2.imread(filepath)

def random_image_subsection(im_filepath):
    """takes a random subsection of my sample image so I can do some testing on image lineing upself."""
    im = Image.open(im_filepath)
    im2 = im.rotate(random.randint(0,359))
    im3 = im2.crop((((im2.height-200)/2),((im2.width-200)/2),im2.height-((im2.height-200)/2),im2.width-((im2.width-200)/2)))
    im3.save("testseg.png")

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
        feature_dict[z] = np.asarray(feature_vec)
        vec_lists.append(np.asarray(feature_vec))
    fin_kp = []
    fin_feat=[]
    for i in feature_dict:
        #fin_kp.append(cv2.KeyPoint(i[0],i[1],0))
        #fin_feat.append(feature_dict[i])
        fin_kp.append(i)
        fin_feat.append(feature_dict[i])
    return fin_kp,feature_dict#np.asarray(fin_feat,np.float32)
    #similarity testing. Using cosign distintances between feature vectors to make sure they are unique enough
    """
    sims = []"/home/mj/catkin_ws/src/StarFinder/l
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

def find_location(img1, img2):
    """Trying this with Orb feature recognition- followed tutorial at link below (copied code too)

    """

    Max_features = 500
    GOOD_MATCH =.15
    #im1_features = Find_Features(im1)
    #im2_features = Find_Features(im2)
    s_kp, s_feat = Find_Features(img1)
    m_kp, m_feat = Find_Features(img2)

    """img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    MIN_MATCH_COUNT = 3

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 500)


    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(s_feat,m_feat,k=2)

    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(s_feat,m_feat,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ s_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ m_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,s_kp,img2,m_kp,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()"""
    # Cosine distances.
    #sims = {}
    #used_pairs = []
    for ad in im1_features[0]:
        nesty_list =[]
        v1 = im1_features[0][ad]
        used_pairs.append((ad,ad))
        for ab in im2_features[0]:
            if (ab,ad) not in used_pairs:
                v2 = im2_features[0][ab]
                nesty_list.append((ab,np.dot(v1,v2)/ (np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))))
                used_pairs.append((ad,ab))
        sims[ad]=(nesty_list.sort(key=lambda x: x[1],reverse=true))
    # now I have a dictionary of the closeness of every point's feature map to any other feature map sorted by
    # i want to assign my feature points to other feature points maximizing the overall match minimizing the difference between the distances of new center_points
    # I should have used classes. ahhhhhh!
    #get two things to match











#Main loop
#Find_Features(image)
random_image_subsection("longExposure.png")
#trying to stitch two ims together
s_1 = cv2.imread("x0_y0_t0.png")
print(s_1.shape)
s_2 = cv2.imread("x_2_y_0_t_0.png")
test_1 = cv2.imread("testseg.png")
find_location(s_1, s_2)
