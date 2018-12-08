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
import yaml

filepath = "/home/mj/catkin_ws/src/StarFinder/longExposure1.png"
image = cv2.imread(filepath)

def random_image_subsection(im_filepath):
    """takes a random subsection of my sample image so I can do some testing on image lineing upself."""
    im = Image.open(im_filepath)
    im2 = im.rotate(random.randint(0,359))
    im3 = im2.crop((((im2.height-200)/2),((im2.width-200)/2),im2.height-((im2.height-200)/2),im2.width-((im2.width-200)/2)))
    im3.save("testseg.png")

def Find_Features(im,save_im=0):
    """This function takes a long exposure image of the star ceiling and finds the feature vectors asssociated with it. """
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # turning the image grayscale
    ret,thresh = cv2.threshold(imgray,127,255,0)
    img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #drawing the bounding contours for stars

    center_points = []
    radi = []
    for i in contours:
        center,radius = cv2.minEnclosingCircle(i)
        center_points.append(center)
        radi.append(radi)
        if save_im == 1:
            cv2.circle(im, (int(center[0]),int(center[1])),int(radius),(0,255,0),-1)
            im[int(center[1]),int(center[0])] = [0,0,255]
    if save_im ==1:
        cv2.imwrite("test_boxes.png", im)

    dist_dict = {}
    #finding the star histagrams
    #i think you need to do angles
    for n in center_points:
        dists = []
        for x in center_points:
            dist = math.sqrt(((x[0]-n[0])**2)+((x[1]-n[1])**2))
            if dist != 0:
                #dists.append(dist)
                dists.append((x,dist))
        dist_dict[n] = dists

    feature_dict = {}
    vec_lists = []
    for z in dist_dict:
        #feature_vec = num stars in... [50,100,200] px and the end is the average dist to the closest 10 stars
        feature_vec = [0,0,0,0,0,0,0,0,0,0,0,0]
        d = dist_dict[z]
        d.sort(key =lambda x: x[1])
        # angle between closest2
        v1=np.array([d[0][0][0]-z[0],d[0][0][1]-z[1]])
        v2=np.array([d[1][0][0]-z[0],d[2][0][1]-z[1]])
        v3=np.array([d[2][0][0]-z[0],d[2][0][1]-z[1]])
        v4=np.array([d[3][0][0]-z[0],d[3][0][1]-z[1]])
        feature_vec[10] = np.dot(v1,v2)/ (np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))
        feature_vec[11]=np.dot(v1,v3)/ (np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v3,v3)))
        #angle between 2nd closest and 3rd closest

        feature_vec[0]= d[0][1]
        feature_vec[1]=d[1][1]
        feature_vec[2]=d[2][1]
        feature_vec[3]=d[3][1]
        feature_vec[4]=d[4][1]
        feature_vec[9]=d[5][1]
        #feature_vec[10]=d[6]
        for r in d:
            if r[1] <= 25:
                feature_vec[6]+=1
                #feature_vec[1]+=1
                #feature_vec[2]+=1
                #feature_vec[3]+=1
            elif r[1] <= 50:
                feature_vec[7]+=1
                #feature_vec[2]+=1
                #feature_vec[3]+=1
            elif r[1] <= 100:
                feature_vec[8]+=1
                #feature_vec[3]+=1
            #elif r <= 200:
            #    feature_vec[9]+=1
        for s in d[0:10]:
            feature_vec[5]+=s[1]
        feature_vec[5] = feature_vec[5]/10
        feature_dict[z] = feature_vec
        vec_lists.append(np.asarray(feature_vec))
    fin_kp = []
    fin_feat=[]
    for i in feature_dict:
        fin_kp.append(cv2.KeyPoint(i[0],i[1],0))
        fin_feat.append(feature_dict[i])
    return fin_kp,np.asarray(fin_feat,np.float32),center_points
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

def find_location(img1, img2,c=0):
    """Trying this with Orb feature recognition- followed tutorial at link below (copied code too)

    """

    Max_features = 500
    GOOD_MATCH =.15
    #im1_features = Find_Features(im1)
    #im2_features = Find_Features(im2)
    s_kp, s_feat,cp_1 = Find_Features(img1)
    m_kp, m_feat,cp_2 = Find_Features(img2)

    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    MIN_MATCH_COUNT = 4

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 30)
    search_params = dict(checks = 1000)


    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(s_feat,m_feat,k=2)

    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(s_feat,m_feat,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.55*n.distance:
            good.append(m)
        #print(m,n)
    print(len(good))
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ s_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #print(src_pts)
        dst_pts = np.float32([ m_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #print (src_pts)
        #print (dst_pts)
        #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        M2, mask2 = cv2.estimateAffinePartial2D(src_pts,dst_pts)

        M2_m = np.array([(M2.tolist()[0][0:2]),(M2.tolist()[1][0:2])])
        M2_a = np.array([[(M2.tolist()[0][2])],[(M2.tolist()[1][2])]])
        #(M2_a)
        matchesMask = mask2.ravel().tolist()
        #print(matchesMask)
        #print(type(M))
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
        #M = np.array(M)
        #print(pts[0])
        #dst = cv2.perspectiveTransform(pts,M)
        dst = pts
        for p in dst:
            p[0] = np.transpose(M2_m.dot(np.asarray([[p[0][0]],[p[0][1]]]))+M2_a)
        nh1 = math.sqrt(((dst[0][0][0]-dst[1][0][0])**2)+((dst[0][0][1]-dst[1][0][1])**2))
        nh2 = math.sqrt(((dst[1][0][0]-dst[2][0][0])**2)+((dst[1][0][1]-dst[2][0][1])**2))
        nh3 = math.sqrt(((dst[2][0][0]-dst[3][0][0])**2)+((dst[2][0][1]-dst[3][0][1])**2))
        nh4 =math.sqrt(((dst[3][0][0]-dst[0][0][0])**2)+((dst[3][0][1]-dst[0][0][1])**2))
        #print((nh1,nh2,nh3,nh4))
        #print(M2_m)
        #print (nh1/nh2)
        if (nh2/w) <= 0.9 or (nh2/w) >= 1.1:
            print("error shrinking the photo to fit")
        #print(nh2/w)
        #print(nh1/h)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,1, cv2.LINE_AA)
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    #print(s_kp)
    img3 = cv2.drawMatches(img1,s_kp,img2,m_kp,good,None,**draw_params)
    #plt.imshow(img3, 'gray'),plt.show()
    # Now I am going to use the image transformation corners to like replot the image on top of the other image

    new_h = max(int(dst[0][0][1]),int(dst[1][0][1]),int(dst[2][0][1]),int(dst[3][0][1]),0,int(h))-min(int(dst[0][0][1]),int(dst[1][0][1]),int(dst[2][0][1]),int(dst[3][0][1]),0,int(h))
    new_w = max(int(dst[0][0][0]),int(dst[1][0][0]),int(dst[2][0][0]),int(dst[3][0][0]),0,int(w))-min(int(dst[0][0][0]),int(dst[1][0][0]),int(dst[2][0][0]),int(dst[3][0][0]),0,int(w))
    r_o = (min(int(dst[0][0][0]),int(dst[1][0][0]),int(dst[2][0][0]),int(dst[3][0][0]),0,int(w)),min(int(dst[0][0][1]),int(dst[1][0][1]),int(dst[2][0][1]),int(dst[3][0][1]),0,int(h)))
    new_im =np.zeros((new_h,new_w, 3))
    #plotting the map image stars in green.
    o_p = []
    n_p = []
    for n in cp_2:
        o_p.append((int(n[1]-r_o[1]),int(n[0]-r_o[0])))

        if int(n[1]-r_o[1])<new_h and int(n[0]-r_o[0]) < new_w:
            if c ==0:
                new_im[int(n[1]-r_o[1]),int(n[0]-r_o[0])]=[255,255,255]
            else:
                new_im[int(n[1]-r_o[1]),int(n[0]-r_o[0])]=[0,255,0]
    for r in cp_1:
        #r = M2.dot(np.asarray([[r[0]],[r[1]],[1.0]]))
        r= (M2_m.dot(np.asarray([[r[0]],[r[1]]])))+M2_a
        n_p.append((int(r[1]-r_o[1]),int(r[0]-r_o[0])))
        #print(r)
        if int(r[1]-r_o[1])<new_h and int(r[0]-r_o[0]) < new_w:
            if c !=0:
                new_im[int(r[1]-r_o[1]),int(r[0]-r_o[0])]=[0,0,255]
            else:
                new_im[int(r[1]-r_o[1]),int(r[0]-r_o[0])]=[255,255,255]
    cv2.imwrite("test_big.png", new_im)
    to_del = []
    #goal: take the op and the np and if the closest point is within a certain range, keep only the original
    filt_im = new_im
    for w in o_p:
        for r in n_p:
            dist = math.sqrt(((w[0]-r[0])**2)+((w[1]-r[1])**2))
            if dist <= 7 and dist != 0:
                to_del.append(r)
    for r in to_del:
        if c !=0:
            filt_im[r[0],r[1]]= [255,0,0]
        else:
            filt_im[r[0],r[1]]= [0,0,0]
    #cv2.imwrite("test_big_filt.png", filt_im)
    cv2.imwrite("fin.png", filt_im)
    return(cv2.imread("fin.png"))






#Main loop
#Find_Features(image)
random_image_subsection("longExposure.png")
#trying to stitch two ims together
s_1 = cv2.imread("x0_y0_t0.png")#
s_7 = cv2.imread("x0_y-3_t0.png")#
s_8 = cv2.imread("x0_y3_t0.png")
#print(s_1.shape)
s_2 = cv2.imread("x_2_y_0_t_0.png")#
s_3 = cv2.imread("x_2_y_-3_t_0.png")#
s_4 = cv2.imread("x_-2_y_-3_t_0.png")#
s_5 = cv2.imread("x_-2_y_0_t_0.png")
s_6 = cv2.imread("x_-2_y_3_t_0.png")
#test_1 = cv2.imread("testseg.png")
pt1=find_location(s_1, s_2)
pt2=find_location(s_3,s_2)
pt3=find_location(pt1,pt2)
pt4 =find_location(s_4,s_7)
pt5 = find_location(s_5,s_4,c=1)
#pt6 = find_location(pt4,pt5)
#pt7 = find_location(pt3,pt6) # v_good
#pt8 = find_location(s_5,pt1,c=1)
