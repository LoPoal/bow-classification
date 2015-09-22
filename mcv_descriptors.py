""" 
	
     Image descriptors
	
     Master in Computer Vision - Barcelona
 
     Week 2 - Task 4: include here the descriptors with color information, such as Opponent SIFT or Color Naming techniques

    Author: Francesco Ciompi, Ramon Baldrich, Jordi Gonzalez
		 
"""

from numpy import *	# numpy, for maths computations
from scipy.ndimage import filters
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots
import cv2
from skimage import feature

# Histogram of Gradient (from 'skimage')
def hog(img):

	S = shape(img)
	HOG = feature.hog(double(img),orientations=9,pixels_per_cell=(S[0]/3,S[1]/3),cells_per_block=(3,3),visualise=True)
	return HOG[0]
