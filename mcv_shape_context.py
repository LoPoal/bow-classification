""" 
    
    SHAPE CONTEXT
    
    Master in Computer Vision - Barcelona
    
    Author: Francesco Ciompi
         
"""

from SC import *
import time
from cv2 import cv
from numpy import *
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots

"""
    
    based on code: https://github.com/creotiv/Python-Shape-Context

"""


#
#    get point from image to apply shape context
# 
def get_points_from_img(src, treshold=50, simpleto=500, t='CANNY'):
    ts = time.time()

    if isinstance(src,str):
        src = cv.LoadImage(src, cv.CV_LOAD_IMAGE_GRAYSCALE)
    test = cv.CreateImage(cv.GetSize(src), 8, 1)

    if t == 'CANNY':
        dst = cv.CreateImage(cv.GetSize(src), 8, 1)
        storage = cv.CreateMemStorage(0)
        cv.Canny(src, dst, treshold, treshold*3, 3)

    A = zeros((cv.GetSize(src)[1],cv.GetSize(src)[0]))
    for y in xrange(cv.GetSize(src)[1]):
        for x in xrange(cv.GetSize(src)[0]):
            A[y,x] = src[y,x]    
    
    px,py = gradient(A)
    points = []
    w,h = cv.GetSize(src)
    for y in xrange(h):
        for x in xrange(w):
            try:
                c = dst[y,x]
            except:
                print x,y
            if c == 255:
                points.append((x,y))
    
    r = 2

    old_lenpoint=len(points)
    counter=0
    while len(points) > simpleto:
        #print len(points)
        '''
        if len(points) == old_lenpoint:
            counter+=1
        else:
            old_lenpoint=len(points)
            counter=0
        if counter>3:
            break
        '''
        newpoints = points
        xr = range(0,w,r)
        yr = range(0,h,r)
        for p in points:
            if p[0] not in xr and p[1] not in yr:
                newpoints.remove(p)
                if len(points) <= simpleto:
                    T = zeros((simpleto,1)) 
                    for i,(x,y) in enumerate(points):
                        T[i] = math.atan2(py[y,x],px[y,x])+pi/2;    
                    return points,asmatrix(T)
        r += 1
    T = zeros((simpleto,1)) 
    for i,(x,y) in enumerate(points):
        T[i] = math.atan2(py[y,x],px[y,x])+pi/2;    
        
    return points,asmatrix(T)





if __name__ == '__main__':

    sco = SC()
    sampls = 10

    points1,t1 = get_points_from_img('000026.jpg',simpleto=sampls)
    #points2,t2 = get_points_from_img('A.png',simpleto=sampls)
    #points3,t3 = get_points_from_img('D.png',simpleto=sampls)
    P1 = sco.compute(points1)
    #P2 = sco.compute(points2)
    #P3 = sco.compute(points3)

    #print P1


    figure(1)
    subplot(1,3,1); imshow(P1)
    #subplot(1,3,2); imshow(P2)
    #subplot(1,3,3); imshow(P3)
    show()