# this file takes the image from the camera, locates all the stars, finds their centers and does some math
import statistics as stat
import os, sys
from os import path
import cv2 as cv2
import numpy as np
import math
import random
import rospkg
from matplotlib import pyplot as plt
from PIL import Image
import yaml

filepath = "/home/lzuehsow/catkin_ws/src/StarFinder/longExposure.png"
image = cv2.imread(filepath)

def random_image_subsection(im_filepath):
    """takes a random subsection of my sample image so I can do some testing on image lineing upself."""
    im = Image.open(im_filepath)
    im2 = im.rotate(random.randint(0,359))
    im3 = im2.crop((((im2.height-200)/2),((im2.width-200)/2),im2.height-((im2.height-200)/2),im2.width-((im2.width-200)/2)))
    im3.save("testseg.png")

def sdot(vec,num):
    """ Square root of dot product """
    return np.sqrt(np.dot(vec[num],vec[num]))

def get_center(im, save_im):
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
    return center_points

def find_features(im,save_im=0):
    """This function takes a long exposure image of the star ceiling and finds the feature vectors asssociated with it. """
    center_points = get_center(im, save_im)
    feat_weights = [2,1,1,1,1,1,1,1,1,1,100,100]
    #feat_weights = [1,1,1,1,1,1,1,1,1,1,1,1]
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
        feature_vec = [0]*12
        d = dist_dict[z]
        d.sort(key =lambda x: x[1])
        # angle between closest2

        v = []
        for i in range(0,4):
            v.append(np.array([d[i][0][0]-z[0],d[i][0][1]-z[1]]))
        feature_vec[10] = np.dot(v[1],v[2])/ (sdot(v,1)*sdot(v,2))
        feature_vec[11]=np.dot(v[1],v[3])/ (sdot(v,1)*sdot(v,3))
        #angle between 2nd closest and 3rd closest

        feature_vec[0]= d[0][1]
        feature_vec[1]=d[1][1]
        feature_vec[2]=d[2][1]
        feature_vec[3]=d[3][1]
        feature_vec[4]=d[4][1]
        feature_vec[9]=d[5][1] # TODO: is feature_vec[9] right?
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
        for i in range(0,len(feature_vec)):
            feature_vec[i] = feature_vec[i]*feat_weights[i]
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
class Feature_Matcher(object):
    def __init__(self, proj_method="partial_affine"):
        self.proj_method = proj_method

    @staticmethod
    def knn_vec_mating(s_feat, m_feat):
        """
        Finds features and uses FLANN to match them.
        """
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 30)
        search_params = dict(checks = 1000)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        return flann.knnMatch(s_feat,m_feat,k=2)

    def find_location(self, img1, img2,c=0,name=""):
        """Stitches the two images together.
        """
        MIN_MATCH_COUNT = 4 #min number of matches to be considered a good match

        #processing the images, finding the key points and centers of all the stars
        s_kp, s_feat, cp_1 = find_features(img1)
        m_kp, m_feat, cp_2 = find_features(img2)

        #KNN FEATURE VECTOR MATING.
        matches = Feature_Matcher.knn_vec_mating(s_feat,m_feat)

        #MATCH FILTERING
        good = []
        for m,n in matches:
            if m.distance < 0.55*n.distance:
                good.append(m)

        print(len(good)) #lets me know how good its doing...

        #if we have enough matches....
        if len(good)>=MIN_MATCH_COUNT:
            #translating from kp indices into coordinates
            src_pts = np.float32([ s_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ m_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M2, matchesMask = self.find_homography(src_pts, dst_pts)
            col_im, new_im = self.find_map_match(img1, img2, M2, (cp_1, cp_2))

            #cv2.imwrite("test_big_filt.png", filt_im)
            cv2.imwrite(name+"fin.png", new_im)
            if c !=0:
                cv2.imwrite(name+"col.png", col_im)
            return(cv2.imread(name+"fin.png"))
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
            return img2

        #code below can draw a picture make sure to uncomment im2 line if used.
        #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,1, cv2.LINE_AA)
        #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       #singlePointColor = None,
                       #matchesMask = matchesMask, # draw only inliers
                       #flags = 2)
        #img3 = cv2.drawMatches(img1,s_kp,img2,m_kp,good,None,**draw_params)
        #plt.imshow(img3, 'gray'),plt.show()

    def find_homography(self, src_pts, dst_pts):
        #ability to control if i am using 3d mapping or not
        if self.proj_method == "partial_affine":
            M2,mask2 = cv2.estimateAffinePartial2D(src_pts,dst_pts)
        elif self.proj_method == "affine":
            M2, mask2 = cv2.estimateAffine2D(src_pts,dst_pts)
        elif self.proj_method == "Homography":
            M2, mask2 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        #FINDING BOUNDS OF NEW IMAGE
        matchesMask = mask2.ravel().tolist()
        return M2, matchesMask

    def find_map_match(self, img1, img2, M2, cp):
        h,w = img1.shape[0:2]
        cp_1, cp_2 = cp
        pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
        nh = []

        #transforming the edges of the images and making sure we are not shrinking it too much
        if self.proj_method in ["partial_affine","affine"]:

            M2_m = np.array([(M2.tolist()[0][0:2]),(M2.tolist()[1][0:2])])
            M2_a = np.array([[(M2.tolist()[0][2])],[(M2.tolist()[1][2])]])

            dst = pts
            for p in dst:
                p[0] = np.transpose(M2_m.dot(np.asarray([[p[0][0]],[p[0][1]]]))+M2_a)
        else:
            dst = cv2.perspectiveTransform(pts,M2)

        nh1 = math.sqrt(((dst[0][0][0]-dst[1][0][0])**2)+((dst[0][0][1]-dst[1][0][1])**2))
        nh2 = math.sqrt(((dst[1][0][0]-dst[2][0][0])**2)+((dst[1][0][1]-dst[2][0][1])**2))
        nh3 = math.sqrt(((dst[2][0][0]-dst[3][0][0])**2)+((dst[2][0][1]-dst[3][0][1])**2))
        nh4 = math.sqrt(((dst[3][0][0]-dst[0][0][0])**2)+((dst[3][0][1]-dst[0][0][1])**2))

        if (nh2/w) <= 0.9 or (nh2/w) >= 1.1:
            print("WARNING shrinking the photo to fit") #warning in case it shrinks it too much to fit
        #new image's height
        dst = np.array(dst).astype(int)

        new_h = max(dst[0][0][1],dst[1][0][1],dst[2][0][1],dst[3][0][1],0,h) \
            -min(dst[0][0][1],dst[1][0][1],dst[2][0][1],dst[3][0][1],0,h)
        #new_image's width
        new_w = max(dst[0][0][0],dst[1][0][0],dst[2][0][0],dst[3][0][0],0,w) \
            -min(dst[0][0][0],dst[1][0][0],dst[2][0][0],dst[3][0][0],0,w)
        #the location of the 2nd image's old origin.
        # TODO: IS THIS SUPPOSED TO BE MIN - MIN?
        r_o = (min(dst[0][0][0],dst[1][0][0],dst[2][0][0],dst[3][0][0],0,w),\
            min(dst[0][0][1],dst[1][0][1],dst[2][0][1],dst[3][0][1],0,h))

        # MAPPING STAR POINTS INTO NEW IMAGE COORDINATE SPACE
        #making an empty image to fill for our final image
        new_im = np.zeros((new_h,new_w, 3))
        col_im = np.zeros((new_h,new_w,3))

        o_p = [] # original_points aka stars in image 2 translated for new image
        n_p = [] # new_points aka stars in image 1
        for pt in cp_2: # mapping the original stars (im2)
            remapped_pt = (int(pt[1]-r_o[1]),int(pt[0]-r_o[0])) # translating the points into new image coordinate system
            o_p.append(remapped_pt) # adding the point to the list of original points
            if remapped_pt[0]<new_h and remapped_pt[1]< new_w: # making sure the point falls in the image
                new_im[remapped_pt[0],remapped_pt[1]]=[255,255,255]
                col_im[remapped_pt[0],remapped_pt[1]]=[0,255,0]

        for ptb in cp_1: #mapping the new stars (im1)
            if self.proj_method in ["partial_affine","affine"]:
                #if the transformation is affine dot the matrix and add the vector
                remapped_ptb = (M2_m.dot(np.asarray([[ptb[0]],[ptb[1]]])))+M2_a
            else:
                #transformation method is homography. just dot matrix.
                remapped_ptb = M2.dot(np.asarray([[ptb[0]],[ptb[1]],[1.0]]))
            #mapping new point to new image coordinate frame
            remapped_ptb = (int(remapped_ptb[1]-r_o[1]),int(remapped_ptb[0]-r_o[0]))
            n_p.append(remapped_ptb) # add the new star to our list of new stars

            if remapped_ptb[0]<new_h and remapped_ptb[1] < new_w: # if star is in the new picture bounds...
                col_im[remapped_ptb[0],remapped_ptb[1]]=[0,0,255]
                new_im[remapped_ptb[0],remapped_ptb[1]]=[255,255,255]

        #DELETING THE REALLY CLOSE POINTS TO REMOVE DUPLICATES
        to_del = []
        CLOSE_VAL = 11 # number of pixels considered "close"

        #goal: take the op and the np and if the closest point is within a certain range, keep only the original
        for old_px in o_p:
            for new_px in n_p:
                dist = math.sqrt(((old_px[0]-new_px[0])**2)+((old_px[1]-new_px[1])**2))
                if dist <= CLOSE_VAL and dist != 0:
                    to_del.append(old_px)
        for pts in to_del:
            col_im[pts[0],pts[1]]= [255,0,0]
            new_im[pts[0],pts[1]]= [0,0,0]
        return col_im, new_im

if __name__ == '__main__':
    #Main loop
    #Find_Features(image)
    rospack = rospkg.RosPack()
    path = str(os.path.join(rospack.get_path('StarFinder'), "photos/"))

    random_image_subsection(path + "longExposure.png")
    pics = ["le_x0y0t0", "le_x2y0t0", "le_x2y-3t0", "le_x-2y-3t0", "le_x-2y0t0", "le_x-2y3t0", "le_x0y-3t0", "le_x0y3t0"]
    s = []
    pt = []

    #trying to stitch two ims together
    for i in range(0,8):
        s.append(cv2.imread(path + pics[i] + ".jpg"))

    fm = Feature_Matcher()
    #fm.proj_method="affine"
    #test_1 = cv2.imread("testseg.png")
    #0,1,2,3,4,5,6,
    pt.append(fm.find_location(s[1], s[0]))#0
    pt.append(fm.find_location(s[2],s[0],c=1))#1
    pt.append(fm.find_location(pt[1],pt[0]))#2
    pt.append(fm.find_location(s[6],s[3]))#3
    pt.append(fm.find_location(pt[2],pt[3]))#4
    pt.append(fm.find_location(s[5],s[0]))#5
    pt.append(fm.find_location(pt[5],pt[4]))#6
    pt.append(fm.find_location(s[4],pt[6]))#7
    pt.append(fm.find_location(pt[7],s[7]))#8
    pt.append(fm.find_location(pt[7],pt[8],c=1))



    #pt3=find_location(pt1,pt2) #v_good
    #pt4 =find_location(s_4,s_7) # v_good
    #pt5 = fm.find_location(s[6],s[3])
    #pt6 = find_location(pt4,pt5)
    #pt7 = find_location(s_5,pt6,c=1)
    #pt7a=find_location(pt5,s_5,c=1)
    #pt7 = find_location(pt3,pt6) # v_good
    #pt8 = find_location(s_5,pt1,c=1)
