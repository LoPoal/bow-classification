""" 
    
    Harris keypoints
    
    Master in Computer Vision - Barcelona
    
    Author: Jan Erik Solem
         
"""

from scipy.ndimage import filters
from numpy import *
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots

"""
    
    Based on the book:
    Jan Erik Solem, 'Programming Computer Vision with Python'
    http://programmingcomputervision.com/

    USAGE
    im = array(Image.open('empire.jpg').convert('L'))
    harrisim = harris.compute_harris_response(im)
    filtered_coords = harris.get_harris_points(harrisim,6)
    harris.plot_harris_points(im, filtered_coords)

"""

def compute_harris_response(im,params):
    """ Compute the Harris corner detector response function for each pixel in a graylevel image. """
    
    sigma = params['harris_sigma']
    
    # derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet / Wtr



def get_harris_points(harrisim,params):
    """ Return corners from a Harris response image min_dist is the minimum number of pixels separating
    corners and image boundary. """

    threshold = params['harris_threshold']
    min_dist = params['harris_min_dist']

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T
    
    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    # sort candidates
    index = argsort(candidate_values)
    
    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
        allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return filtered_coords


def plot_harris_points(image,filtered_coords):
    """ Plots corners found in image. """

    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
    axis('off')
    show()


def get_descriptors(image,filtered_coords,params):
    """ For each point return pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid). """

    wid = params['harris_wid']
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
        coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
        
    return desc
    


