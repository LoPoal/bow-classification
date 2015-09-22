"""
    
    Bag of Word (BoW) Tools
    
    Master in Computer Vision - Barcelona, 2015
    
    Authors: Jordi Gonzalez, Ramon Baldrich, Eduard Ramon, Cristhian Aguilera, Miquel Ferrarons

         
"""

# Master in Computer Vision Libraries
import mcv_dsift as dsift            # dense SIFT [cite source]
import mcv_sift as sift                # alternative code from book
from mcv_descriptors import *
from color_constancy import * #Color constancy algorithms
import mcv_harris as harris
import mcv_shape_context
import SC
import mcv_centrist
import mcv_gist
from ColorNaming import *
from sklearn.decomposition import PCA

# External Libraries
import os                            # operating system
from numpy import *                    # numpy, for maths computations
from PIL import Image                 # python image library
import random as rnd                # random value generator
import cv2.cv as cv                            # opencv
import cv2
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots
import logging
from skimage.feature import local_binary_pattern
import csv

logger_tools = logging.getLogger('log')

#
#     get list of files in 'path' with extension 'ext'
#
def getFilelist(path,ext='',queue=''):
    # returns a list of filenames (absolute, relative) for all 'ext' images in a directory
    if ext != '':
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]    


#
#    detect random points in the image domain
#
def detectPoints(imgSize,nPt,mode='rnd'):    
    """ compute the position of 'nPt' points in the image domain 'dataSize' with modality 'mode' 
    if 'dense' or a specific detector is declared, nPt is bypassed """
    
    if mode == 'rnd':
        return rnd.sample(range(0,imgSize[0]*imgSize[1]),  nPt);    
    else:
        logger_tools.info('Unkwnown Detection Mode!')
        return None

#
#     random subsample over the rows of data matrix 'data'
#
def randSubsample(data, nPt):

    #logger_tools.info( 'Shape of data : '+ shape(data))
    #logger_tools.info( shape(data)[0] )# = 502
    #logger_tools.info( 'number of points:'+ str(nPt) )
    if shape(data)[0] > nPt:
        idx = array(rnd.sample(range(0,shape(data)[0]), nPt))
        return data[idx]
    else:
        return data

#    
#     compute feature matrix per image
#
def getFeatImg(img, params, showFig=False, filename=''):
    """     Compute features corresponding to the selcted 'descriptor' for image 'img'
    The descriptor is applied to patches (patchSize) over a dense matrix (gridSpacing) 
    in case of SIFT and HOG, and to keypoints in all the others. It also implements the
    early fusion as a combination of descriptors """

    #-------------------------------------------------------------------------------------------------------------------
    # Our implementation of getFeatImg. Using OpenCV
    #-------------------------------------------------------------------------------------------------------------------

    #img = cv2.imread(filename)
    # Create detector
    # ["FAST","STAR","SIFT","SURF","ORB","MSER","GFTT","HARRIS","Dense"]
    opencvdic = {'fast': 'FAST', 'star': 'STAR', 'sift': 'SIFT', 'mser': 'MSER', 'gftt': 'GFTT',
                 'harris': 'HARRIS', 'surf': 'SURF', 'dense': 'Dense', 'orb': 'ORB'}
    detector = cv2.FeatureDetector_create(opencvdic[params['keypointDetector']])
    # find the keypoints with STAR
    kp = detector.detect(img, None)

    # if showFig:
    #     plt.figure(params['keypointDetector'])
    #     plt.imshow(img)
    #     plt.hold(True)
    #     for keyp in kp:
    #         x, y = keyp.pt
    #         plt.plot(x, y, 'ro')
    #         plt.axis('equal')
    #     plt.show()

    # LOCAL DESCRIPTORS

    if len(kp) < 1:
        logger_tools.info( 'Image skipped due to a lack of key-points')
    else:
        DESCRIPTORS = []
        POSITIONS = []
        # ####################################    
        #     Compute image descriptor(s)    
        # ####################################
        #
        #     Include (optional) early fusion
        #
        # ####################################
        nf = get_num_of_features(params)

        #if params['keypointDetector'] != 'void': # if not image retrieval
        #    DESCRIPTORS = zeros((len(kp), nf))

        descrPos = 0
        opencvdic = {'brief': 'BRIEF', 'surf': 'SURF', 'sift': 'SIFT', 'hog': 'HOG', 'orb': 'ORB', 'freak': 'FREAK', 'centrist': 'CENTRIST', 'gist': 'GIST', 'lbp': 'LBP'}

        desnumber = 1
        for descriptor in params['descriptor']:
            # ##############################################    
            #     SIFT, SURF, SIFT, BRIEF, ORB, FREAK
            # ##############################################

            desName= descriptor.replace("Opponent","")

            if desName in ['sift', 'brief', 'surf', 'orb', 'freak']:


                if "Opponent" in descriptor:
                    opencvDescriptorName= "Opponent"+opencvdic[desName]
                else:
                    opencvDescriptorName= opencvdic[desName]

                #logger_tools.info('Computing the generic OpenCV '+opencvDescriptorName+' descriptor')
                descriptor_extractor = cv2.DescriptorExtractor_create(opencvDescriptorName)

                if desName == 'surf':
                    descriptor_extractor.setInt("extended", 1)

                if desnumber == 1:
                    kp, des = descriptor_extractor.compute(img, kp)
                    DESCRIPTORS = des
                else:
                    new_keypoints = []
                    new_des = []

                    #For each descriptor
                    for i in range(0, len(kp)):
                        kpp, des = descriptor_extractor.compute(img, [kp[i]])
                        if len(kpp) > 0:
                            new_keypoints.append(kp[i])
                            tmp = (DESCRIPTORS[i].tolist() + des[0].tolist())
                            new_des.append(np.array(tmp))
                    DESCRIPTORS=new_des
                    kp=new_keypoints
                desnumber += 1
            
            # ########################################
            #     Histogram of Oriented Gradient (HOG)    
            # ########################################
            elif descriptor == 'hog':
                #logger_tools.info( 'Computing HOG of '+str(params['nf_hog'])+' elems')
                new_keypoints = []
                new_des = []
                grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if len(kp) > 0:
                    for i in range(0, len(kp)):
                        r = kp[i].pt[0]
                        c = kp[i].pt[1]
                        r = int(r)
                        c = int(c)

                        patch = grayimage[r-int(50/2):r+int(50/2), c-int(50/2):c+int(50/2)]
                        if size(patch) == 50*50:  # discard incomplete patches
                            # hog is found in mcv_descriptors.py
                            #des = hog(patch).transpose()
                            des = hog(patch).transpose()
                            new_keypoints.append(kp[i])
                            if desnumber == 1:
                                new_des.append(des)
                            else:
                                tmp=(DESCRIPTORS[i].tolist() + des.tolist())
                                new_des.append(np.array(tmp))

                DESCRIPTORS = np.array(new_des)
                kp = new_keypoints
                desnumber += 1

            # ########################################
            #     LBP
            # ########################################
            elif descriptor == 'lbp':
                #logger_tools.info('Computing LBP of '+str(params['nf_lbp'])+' elems.')
                new_keypoints = []
                new_des = []
                grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if len(kp) > 0:
                    for i in range(0, len(kp)):
                        r = kp[i].pt[0]
                        c = kp[i].pt[1]
                        r=int(r)
                        c=int(c)

                        patch = grayimage[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                        if size(patch) == params['lbp_patch_size']*params['lbp_patch_size']: # discard incomplete patches

                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')

                            des,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            new_keypoints.append(kp[i])
                            if desnumber == 1:
                                new_des.append(des)
                            else:
                                tmp=(DESCRIPTORS[i].tolist() + des.tolist())
                                new_des.append(np.array(tmp))

                DESCRIPTORS = np.array(new_des)
                kp = new_keypoints
                desnumber += 1
            # ########################################
            #     LBP COLOR
            # ########################################
            elif descriptor == 'lbp_color':
                #logger_tools.info('Computing LBP COLOR of '+str(params['nf_lbp_color'])+' elems')
                new_keypoints = []
                new_des = []
                grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if len(kp) > 0:
                    for i in range(0, len(kp)):
                        r = kp[i].pt[0]
                        c = kp[i].pt[1]
                        r=int(r)
                        c=int(c)

                        patch = grayimage[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                        if size(patch) == params['lbp_patch_size']*params['lbp_patch_size']: # discard incomplete patches
                            blue, green, red = cv2.split(img)

                            patch = blue[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')
                            desb,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            patch = green[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')
                            desg,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            patch = red[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')
                            desr,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            des = np.hstack((desb,desg,desr))
                            new_keypoints.append(kp[i])
                            if desnumber == 1:
                                new_des.append(des)
                            else:
                                tmp=(DESCRIPTORS[i].tolist() + des.tolist())
                                new_des.append(np.array(tmp))

                DESCRIPTORS = np.array(new_des)
                kp = new_keypoints
                desnumber += 1

            # ####################################            
            #     Week 1 - Task 3: Shape Context (SC)
            # ####################################
            elif descriptor == 'sc':
                
                #print 'Shape Context ('+str(params['nf_sc'])+' elems.)'
                sco = SC.SC()
                #print img.shape
                points1,t1 = mcv_shape_context.get_points_from_img(filename)
                descriptors = sco.compute(points1)
                DESCRIPTORS = descriptors
                #pass
                
            # ####################################            
            #     Week 2 - Task 3: CENTRIST        
            # ####################################
            elif descriptor == 'centrist':
                #logger_tools.info( 'Computing CENTRIST of '+str(params['nf_centrist'])+' elems.')
                
                #Returns the histogram, and the censusTransformedImage. We keep the histogram as descriptor
                grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist, ctImage  = mcv_centrist.centrist(grayimage)
                new_des = []
                feats = np.array(hist)
                
                if desnumber == 1:
                    new_des.append(feats)
                else:
                    new_des = np.append(DESCRIPTORS,feats)                                        
                    
                DESCRIPTORS = np.array(new_des)
                desnumber += 1
                            
                
            elif descriptor == 'gist':
                #logger_tools.info( 'Computing GIST of '+str(params['nf_gist'])+' elems.')
                des = mcv_gist.gist(img)
                feats= np.array(des)
                if desnumber == 1:
                    new_des.append(feats)
                else:
                    new_des = np.append(DESCRIPTORS,feats)                                        
                    
                DESCRIPTORS = np.array(new_des)
                desnumber += 1
 

            # ####################################
            # Color naming
            # ####################################
            elif descriptor == 'color_naming':
                #logger_tools.info( 'Color_Naming ('+str(params['nf_cc'])+' elems.)')
                new_keypoints = []
                new_des = []
                #grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if len(kp) > 0:
                    for i in range(0, len(kp)):
                        r = kp[i].pt[0]
                        c = kp[i].pt[1]
                        r = int(r)
                        c = int(c)


                        patch = img[r-int(60/2):r+int(60/2), c-int(60/2):c+int(60/2),:]
                        if size(patch) == 60*60*3: # discard incomplete patches
                            # cc is found in color_naming.py
                            #TO DO des = ...
                            des = getColorNamingDescriptor( patch )
                            new_keypoints.append(kp[i])
                            if desnumber == 1:
                                new_des.append(des)
                            else:
                                tmp=(DESCRIPTORS[i].tolist() + des.tolist())
                                new_des.append(np.array(tmp))

                DESCRIPTORS = np.array(new_des)
                kp = new_keypoints
                desnumber += 1
        
    POSITIONS = getPositions(kp)

    return DESCRIPTORS, POSITIONS



#
#     normalize feature matrix
#
def normalizeFeat(x,mean_x=None,std_x=None):
    
    if mean_x == None and std_x is None:
        mean_x = x.mean(axis=0)
        std_x = x.std(axis=0)
        std_x[std_x==0] = 1
                
    return (x-tile(mean_x,(shape(x)[0],1)))/tile(std_x,(shape(x)[0],1)),mean_x,std_x
    
    
#    
#     accuracy of a confusion matrix    
#
def accuracy(confMat):
    
    return sum(diag(confMat))/sum(confMat.flatten())
    
    
#    
#     compute the row with minimum distance between vector 'v' and matrix 'm'
#
def yMinDist(v,m,metric):

    Dist = zeros((1,shape(m)[0]))
    
    for row in range(0,shape(m)[0]):
        
        if metric == 'euclidean':

            Dist[0,row] = sqrt(sum((m[row,:]-v)**2))
    #print 'Distances to the clusters:'
    #print Dist
    return argmin(Dist[0,:])

def yMinDist_andDist(v,m,metric):

    Dist = zeros((1,shape(m)[0]))

    for row in range(0,shape(m)[0]):

        if metric == 'euclidean':

            Dist[0,row] = sqrt(sum((m[row,:]-v)**2))

    return argmin(Dist[0,:]),min(Dist[0,:])
        
#
#    Hamming distance
#
""" Calculate the Hamming distance between two given strings """
def hamming(a, b):
    
    return sum(logical_xor(a,b).astype(int))

    
    
#
#    show first 10 results of image-based query
#    
def showQueryResult(filename,trainDir,LIBRARY,idxs,method,relevance_feedback=False):
            
    #
    #    QUERY RESULTS VISUALIZATION (first 10 results)
    #                    
    fig = figure('IMAGE RETRIEVAL'); subplot(353); title('QUERY')
    img = array(Image.open(filename))
    imshow(img,cmap=gray)

    subplot(3,5,6)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[0]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[0]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
        
    subplot(3,5,7)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[1]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[1]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
        
    subplot(3,5,8)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[2]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[2]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
                
    subplot(3,5,9)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[3]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[3]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
                
    subplot(3,5,10)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[4]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[4]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
            
    subplot(3,5,11)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[5]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[5]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
            
    subplot(3,5,12)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[6]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[6]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
                
    subplot(3,5,13)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[7]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[7]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
            
    subplot(3,5,14)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[8]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[8]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
                
    subplot(3,5,15)
    try:
        img = array(Image.open(''+trainDir+'car/'+LIBRARY[idxs[9]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
    except:
        img = array(Image.open(''+trainDir+'dog/'+LIBRARY[idxs[9]].split('/')[-1][0:-4]+'JPEG')); imshow(img)
                
    
    if relevance_feedback:
                
        mouse = MouseMonitor()
        mouse.weights = []
        mouse.cont = 0
        connect('button_press_event', mouse.mycall)
        
        show()

        weights = mouse.weights
        
        del(mouse)    

        return weights

    else:
        show()

def get_num_of_features(params):
    nf = 0

    for descriptor in params['descriptor']:
        nf = nf + get_num_of_features_descriptor(descriptor, params)

    return nf

def get_num_of_features_descriptor(descriptor, params):
    nf = 0
    originalDescriptorName = descriptor
    if "Opponent" in originalDescriptorName:
        descriptor= descriptor.replace("Opponent","")
    if descriptor == 'sift':
        nf = nf + params['nf_sift']
    if descriptor == 'hog':
        nf = nf + params['nf_hog']
    if descriptor == 'surf':
        nf = nf + params['nf_surf']
    if descriptor == 'sc':
        nf = nf + params['nf_sc']
    if descriptor == 'brief':
        nf = nf + params['nf_brief']
    if descriptor == 'freak':
        nf = nf + params['nf_freak']
    if descriptor == 'orb':
        nf = nf + params['nf_orb']
    if descriptor == 'centrist':
        nf = nf + params['nf_centrist']
    if descriptor == 'gist':
        nf = nf + params['nf_gist']
    if descriptor == 'lbp':
        nf = nf + params['nf_lbp']
    if descriptor == 'color_naming':
        nf = nf + params['nf_cc']
    if descriptor == 'lbp_color':
        nf = nf + params['nf_lbp_color']


    if "Opponent" in originalDescriptorName:
        if descriptor in ['sift', 'brief', 'surf', 'orb', 'freak']:
            nf= nf*3
        else:
            logger_tools.info( 'ERROR: Opponent variants just supported for SIFT,BRIEF,SURF,ORB and FREAK descriptors\033[0m')
            exit(-1)

    return nf

def getPositions(keypoints):
    pos=[]
    for k in keypoints:
        pos.append(k.pt)
    return pos

def apply_color_constancy(img, bowParams):
    """
    Method to aply color constancy
    """
    if bowParams['cc'] != 'no':
        if bowParams['cc'] == 'grey_world':
            img = grey_world(img)
        elif bowParams['cc'] == 'retinex':
            img = retinex(img)
        elif bowParams['cc'] == 'max_white':
            img = max_white(img)
        elif bowParams['cc'] == 'retinex_adjust':
            img = retinex_adjust(img)
    return img



def store_results_on_csv(filename, params, accurasy, confusionMatrix):
    """
    Method to store the results
    """
    params_to_store= params.copy()
    params_to_store['accurasy'] = accurasy
    conf=str(confusionMatrix.tolist()).replace(',', '')
    params_to_store['confusionMatrix'] = conf

    #Store header
    with open(filename+'header.csv', 'wb') as f:  # Just use 'w' mode in 3.x
        f.write( str(params_to_store.keys()) )


    #Store results
    with open(filename+'.csv', 'a') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, params_to_store.keys())
        w.writerow(params_to_store)

class MouseMonitor():

    event = None
    xdatalist = []
    ydatalist = []
    weights = []
    cont = 0

    def mycall(self, event):

        self.event = event
        self.xdatalist.append(event.x)
        self.ydatalist.append(event.y)
        self.cont+=1

        if event.button == 1:
            self.weights.append(1)
            logger_tools.info( 'result '+str(self.cont)+'. RELEVANT')
        elif event.button == 3:
            self.weights.append(-1)
            logger_tools.info( 'result '+str(self.cont)+'. NOT RELEVANT')

