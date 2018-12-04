#image processing but with classes ooooh ahhhhhh

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

class Star(object):
    """ An individual star point"""
    def _init_ (self,args):
        self.x = 0 # x position in Star_Field
        self.y = 0 # y position in Star_Field
        self.top_matches = [] # top_likely points

class Star_Field(object):
    """ A bunch of Stars"""
    def _init_ (self,args):
        
