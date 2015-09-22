""" 
    
    CENTRIST descriptor
    
    Master in Computer Vision - Barcelona
    
    Based on Matlab implementation
    Author: Sebastian Boehm
    Python code by: Francesco Ciompi
         
"""


from numpy import *
from PIL import Image
from scipy import *
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots
import cv2

def censusTransformImage(img):
    
    # y-1 x-1
    censusTransformedImage = zeros((shape(img[1:-1,1:-1])))
    censusTransformedImage[img[1:-2,1:-2]>=img[:-3,:-3]] = 1
        
    ## y-1 x
    tmp = zeros((shape(img[1:-1,1:-1])))
    tmp[img[1:-1,1:-1]>=img[:-2,1:-1]] = 2
    censusTransformedImage = censusTransformedImage + tmp
    
    ## y-1 x+1
    tmp = zeros((shape(img[1:-1,1:-1])))
    tmp[img[1:-1,1:-1]>=img[:-2,2:]] = 4
    censusTransformedImage = censusTransformedImage + tmp
    
    ## y x-1
    tmp = zeros((shape(img[1:-1,1:-1])))
    tmp[img[1:-1,1:-1]>=img[1:-1,:-2]] = 8
    censusTransformedImage = censusTransformedImage + tmp
    
    ## y x+1
    tmp = zeros((shape(img[1:-1,1:-1])))
    tmp[img[1:-1,1:-1]>=img[1:-1,2:]] = 16
    censusTransformedImage = censusTransformedImage + tmp
    
    ## y+1 x-1
    tmp = zeros((shape(img[1:-1,1:-1])))
    tmp[img[1:-1,1:-1]>=img[2:,:-2]] = 32
    censusTransformedImage = censusTransformedImage + tmp
    
    ## y+1 x
    tmp = zeros((shape(img[1:-1,1:-1])))
    tmp[img[1:-1,1:-1]>=img[2:,1:-1]] = 64
    censusTransformedImage = censusTransformedImage + tmp
    
    ## y+1 x+1
    tmp = zeros((shape(img[1:-1,1:-1])))
    tmp[img[1:-1,1:-1]>=img[2:,2:]] = 128
    censusTransformedImage = censusTransformedImage + tmp
    
    return censusTransformedImage

#Receive directly the grayscale image
def centrist(grayImage):        
        
    ctImage = censusTransformImage(grayImage)
    
    hist, binEdges = histogram(ctImage.flatten(), density=True)
            
    ## drop first and last column in histogram
    censusTransformHistogram = hist[1:-1]
    
    return hist, ctImage