""" 
    
    SIFT
    
    Master in Computer Vision - Barcelona
    
    Author: Francesco Ciompi, Ramon Baldrich, Jordi Gonzalez
         
"""

from PIL import Image
import os
import platform
from numpy import *
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots

"""

    Based on:
    
    [Lowe04] Lowe, D. G., "Distinctive Image Features from Scale-Invariant Keypoints", International Journal of Computer Vision, 
    60, 2, pp. 91-110, 2004.
    
    [book] Programming Computer Vision with Python (http://programmingcomputervision.com/)     

"""


def process_image(imagename,resultname,params):
# """ Process an image and save the results in a file. """

    params_string = "--edge-thresh "+str(params['sift_edge-thresh'])+" --peak-thresh "+str(params['sift_peak-thresh'])

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
    
    # path to the SIFT code
    sift_bin = './sift_code/sift_linux'
    if platform.system() == 'darwin':
        sift_bin = './sift_code/sift'
    elif platform.system() == 'win32':
        sift_bin = './sift_code/sift_win'

    cmmd = str(sift_bin+' '+imagename+" --output="+resultname+" "+params_string)
    os.system(cmmd)



def read_features_from_file(filename):
# Read features properties and return in matrix form

    f = loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors


def write_features_to_file(filename,locs,desc):
# Save features location and descriptor to file
    
    savetxt(filename,hstack((locs,desc)))
    


def plot_features(im,locs,circle=False):
# Show image with features. input: im (image as array),
# locs (row, col, scale, orientation of each feature).

    def draw_circle(c,r):
        t = arange(0,1.01,.01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x,y,'b',linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
    else:
        plot(locs[:,0],locs[:,1],'ob')
    axis('off')


    
