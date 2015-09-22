""" 
    
    Bag of Word (BoW) library
    
    Master in Computer Vision - Barcelona, 2015
    
    Authors: Jordi Gonzalez, Ramon Baldrich, Eduard Ramon, Cristhian Aguilera, Miquel Ferrarons

         
"""
from sklearn.decomposition import PCA
from mcv_svm_roc import *        # mcv module for SVM ROC curve
from mcv_svm import *            # mcv module for SVM
from mcv_tools import *            # mcv personal library of tools
from mcv_kmeans import *            # mcv personal library of tools


import time
#from numpy import *                # numpy, for maths computations
import numpy as np
import random as rnd            # random library
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots
import pickle                    # save/load data files
from scipy.cluster.vq import *    # for k-means
from sklearn import decomposition   # for PCA
from sys import exit    
import skimage
import platform
import multiprocessing as mp
import logging
from sklearn.cluster import KMeans

logger_bow = logging.getLogger('log')

#
#     compute image descriptors
#
def getDescriptor(trainFiles, outputDir, bowParams, showFig, runmultiprocess):
    
    # the features subsampling is done before storing, to save space!
    logger_bow.info('Starting multiprocesses for computing descriptors')
    
    if runmultiprocess: pool = mp.Pool(mp.cpu_count() - 1 if mp.cpu_count() > 1 else 0)
        
    iteracio = 1
    for filesList in trainFiles:
        #logger_bow.info('Starting computing descriptors for class '+str(iteracio)+' of '+str(len(trainFiles)))
        for fileName in filesList:
            if runmultiprocess:
                pool.apply_async(getDescriptorImage, args=(fileName, outputDir, bowParams, False))
            else:
                getDescriptorImage(fileName, outputDir, bowParams, showFig)
        
        #logger_bow.info('Finishing computing descriptors for class '+str(iteracio)+' of '+str(len(trainFiles)))
        iteracio = iteracio + 1
            
    if runmultiprocess:
        pool.close()
        pool.join()
    logger_bow.info('Multiprocesses for computing descriptors finished')


#
#     compute one single  image descriptors
#
def getDescriptorImage(fileName, outputDir, bowParams, showFig):

    img = cv2.imread(fileName)
    img = apply_color_constancy(img, bowParams)

    if bowParams['fusion'] == 'early':

        # getFeatImg can be found at: mcv_tools.py
        feats, pos = getFeatImg(img, bowParams, showFig, fileName)

        if feats is not None:
            # randSubsample can be found at: mcv_tools.py
            descr = ''
            for d in bowParams['descriptor']:
                descr = ''+descr+d+'_'
            descr = descr[0:-1]
            f = open(''+outputDir+''+fileName.split('/')[-1][0:-4]+'_'+bowParams['keypointDetector']+'_'+descr+'.pkl',
                     'wb')
            pickle.dump(feats, f)
            f.close()
            
    elif bowParams['fusion'] == 'late' or bowParams['fusion'] == 'off':
        
        for descriptor in bowParams['descriptor']:

            params = bowParams.copy(); params['descriptor'] = [''+descriptor+'']
            feats,pos = getFeatImg(img,params,showFig,fileName);
            if feats is not None:
                feats = randSubsample(feats, bowParams['nPt'])
                f = open(''+outputDir+''+fileName.split('/')[-1][0:-4]+'_'+bowParams['keypointDetector']+'_' +
                         descriptor+'.pkl', 'wb')
                pickle.dump(feats, f)
                f.close()
            logger_bow.info(fileName)


#
#     compute vocabulary from image descriptors
#
def getVocabulary(inDir, ext, classifierDir, bowParams, showFig=False):
    if bowParams['fusion'] == 'late':
        nv = len(bowParams['descriptor'])  # number of vocabularies
    else:
        nv = 1

    for voc in range(nv):
        if bowParams['fusion'] == 'early':

            descr = ''
            for d in bowParams['descriptor']:
                descr = ''+descr+d+'_'
            descr = descr[0:-1]

            nf = get_num_of_features(bowParams)

        else:

            descr = bowParams['descriptor'][voc]
            nf= get_num_of_features_descriptor(descr, bowParams)

        #logger_bow.info('Checking the number of descriptors...')
        nd = getNumDescriptor(inDir,ext,''+bowParams['keypointDetector']+'_'+descr+'')  # number of collected descriptors
        #logger_bow.info('found '+str(nd)+' descriptors')

        # initialize matrix
        #logger_bow.info( 'Initialize descriptors matrix...'  )
        X = zeros((nd, nf))
            
        files, null = getFilelist(inDir,ext,''+bowParams['keypointDetector']+'_'+descr+'')
        #logger_bow.info(''+bowParams['keypointDetector']+'_'+descr+'')
        ridx = 0
        #logger_bow.info( 'collecting descriptors')
        for file in files:
            #logger_bow.info('Loading '+file)
            with open(file, 'rb') as f:
                    feats = pickle.load(f)
                    s = shape(feats)
                    #logger_bow.info( s)
                    X[ridx:ridx+s[0], :] = feats
                    ridx = ridx + s[0]
                    
                    if showFig:
                        figure(2)
                        imshow(feats)    
                        draw()                    


        # normalize D matrix: normalizeFeat is found at mcv_tools.py
        X, meanX, stdX = normalizeFeat(X)

        #MIQUEL PCA
        if(bowParams['nf_pca']>0):
            nf_pca= bowParams['nf_pca']
            pca = decomposition.PCA(nf_pca)
            pca.fit(X)
            X = pca.transform(X)
            filepath= classifierDir+bowParams['keypointDetector']+'_PCA'+str(nf_pca)+'.pkl'
            f = open(filepath, 'wb')
            logger_bow.info( "Storing PCA features in " + filepath)
            pickle.dump(pca,f)
        #END MIQUEL PCA

        # k-means
        logger_bow.info( 'Starting clustering for vocabulary computation...')
        
        ts = time.time()
        
        centroids,variance = mcv_kmeans(X,bowParams['K'],5,1e-3,multiprocess=-2)
        
        # Alternatives:
        #1) km = KMeans(n_clusters=bowParams['K'], init='k-means++', max_iter=5, tol=1e-3, n_jobs=-1)
        #cen=km.fit(X)
        #centroids=cen.cluster_centers_
        
        #2) centroids, variance = kmeans(X, bowParams['K'], 5, 1e-3)

        te = time.time()
        logger_bow.info('Clustering finished in %2.2f sec' % (te-ts))
        
        code,distance = vq(X,centroids)    # Assigns a code from a code book to each observation. 
        
        filepath = classifierDir+bowParams['keypointDetector']+'_'+descr+'_centroids.pkl'
        f = open(filepath, 'wb')
        pickle.dump(centroids,f)
        pickle.dump(meanX,f)
        pickle.dump(stdX,f)
        f.close()  # store centroids
        logger_bow.info( 'Codebook stored in ' + filepath)
        if nv > 1:
            logger_bow.info( 'CLEANING variables for the other vocabulary...')
            meanX= None
            stdX= None
            X= None

#
#     assign words of vocabulary to each image
#
def getAssignment(inDir,trainLabels,classifierDir,trainDataDir,bowParams,showFig, runmultiprocess):    

    #MIQUEL PCA
    pca= None
    if(bowParams['nf_pca']>0):
        nf_pca= bowParams['nf_pca']
        filepath= classifierDir+bowParams['keypointDetector']+'_PCA'+str(nf_pca)+'.pkl'
        f = open(filepath, 'rb')
        logger_bow.info( 'Opening PCA features from '+filepath)
        pca =  pickle.load(f)
    #END MIQUEL PCA
            
    if bowParams['fusion'] == 'late':
        nv = len(bowParams['descriptor']) # number of vocabularies
    else:
        nv = 1

    for voc in range(nv):

        if bowParams['fusion'] == 'early':

            descr = ''
            for d in bowParams['descriptor']: descr = ''+descr+d+'_'
            descr = descr[0:-1]

            nf = get_num_of_features(bowParams)
            params = bowParams.copy()
        else:
            descr = bowParams['descriptor'][voc]
            nf= get_num_of_features_descriptor(descr, bowParams)
            params = bowParams.copy(); params['descriptor'] = [''+descr+'']  # update descriptor for late fusion
    
        nh = 1
        if bowParams['spyramid']:
            nh=get_num_hist_pyramid(bowParams['pyrconf'])
            
        # initialize signatures (if knn)
        if bowParams['classifier'] == 'knn':
            signatures = zeros((int(max(trainLabels)), nh*bowParams['K']))
            nPoints_per_class = zeros((int(max(trainLabels)), 1))
            for i in range(0,int(max(trainLabels))):
                nPoints_per_class[i] = sum(trainLabels == i+1)    
            nPoints_per_class[nPoints_per_class == 0] = 1
                
        # prepare training data (if svm) 
        if bowParams['classifier'] == 'svm':
            trainData = zeros((len(trainLabels), nh*bowParams['K']))
                
        f = open(classifierDir+bowParams['keypointDetector']+'_'+descr+'_centroids.pkl', 'rb')
        global meanX
        global stdX
        global centroids 
        centroids = pickle.load(f)
        meanX = pickle.load(f)
        stdX = pickle.load(f)
        f.close()

        count = 0
        
        logger_bow.info( 'Starting the assignment of words')
        if runmultiprocess: 
            pool = mp.Pool(mp.cpu_count()- 1 if mp.cpu_count()>1 else 0)
            PoolResults = []
        idx_img = 0        
        for filesList in inDir:
            for fileName in filesList:
                if bowParams['classifier'] == 'knn':
                    idx = trainLabels[idx_img]
                if bowParams['classifier'] == 'svm':
                    idx = idx_img                    

                if runmultiprocess:
                    if bowParams['fusion']=='late':
                        PoolResults.append(pool.apply_async(getAssignmentImage, args=(fileName, idx, params, False, pca)))
                    else:
                        PoolResults.append(pool.apply_async(getAssignmentImage, args=(fileName, idx, bowParams, False, pca)))
                else:
                    if bowParams['fusion']=='late':
                        signature,idx = getAssignmentImage(fileName, idx, params, showFig,pca)
                    else:
                        signature,idx = getAssignmentImage(fileName, idx, bowParams, showFig,pca)
                    idx = int(idx)
                    if bowParams['classifier'] == 'knn':
                        signatures[idx-1, :] = signatures[idx-1, :] + signature/nPoints_per_class[idx-1]
                    if bowParams['classifier'] == 'svm':
                        trainData[idx,:] = signature                    
                idx_img += 1
                
        if runmultiprocess:
            while (len(PoolResults)>0):
                try:
                    signature,idx = PoolResults[0].get(timeout=0.001)
                    PoolResults.pop(0)
                    if bowParams['classifier'] == 'knn':
                        signatures[idx-1,:] = signatures[idx-1,:] + signature/nPoints_per_class[idx-1]
                    if bowParams['classifier'] == 'svm':
                        trainData[idx,:] = signature                    
                except:
                    pass
                
            pool.close()
            pool.join()

        # store training data
    
        # knn
        if bowParams['classifier'] == 'knn':
            f = open(trainDataDir+descr+'_'+bowParams['keypointDetector']+'_bow_signatures_K_'+str(bowParams['K'])+'.pkl', 'wb')
            pickle.dump(signatures,f);
            f.close()        
    
        # svm
        if bowParams['classifier'] =='svm':
            f = open(trainDataDir+descr+'_'+bowParams['keypointDetector']+'_svm_traindata_K_'+str(bowParams['K'])+'.pkl', 'wb')
            pickle.dump(trainData,f);
            pickle.dump(trainLabels,f);
            f.close()        
        logger_bow.info( 'Assignment finished!')

#
#     assign words of vocabulary to a single image
#

def getAssignmentImage(fileName,Id,params,showFig,pca=None):
    #logger_bow.info( fileName)

    imgRGB = array(Image.open(fileName))
    img = cv2.imread(fileName)
    img = apply_color_constancy(img, params)
  
    # getFeatImg is found at mcv_tools.py
    feats,pos = getFeatImg(img,params, filename=fileName)

    # normalizeFeat is found at mcv_tools.py
    feats = normalizeFeat(feats,meanX,stdX)[0]

    #MIQUEL PCA
    if params['nf_pca'] > 0:
        feats = pca.transform(feats)
    #END MIQUEL PCA
    code,distance = vq(feats,centroids) # nearest-neighbour            

    # Week3: use of spatial pyramids
    if params['spyramid']:                    
        S = shape(img)
        hc = codeHist(code,params['K'],params['spyramid'],params['pyrconf'],S,pos,img,showFig)
    else:
        hc = codeHist(code,params['K'])                

    # cumulate normalized descriptors in 'signatures'
    
    
    signature = hc

        
    if showFig:
        ion()
        figure('Words assignment')
        subplot(2,1,1)
        imshow(hstack((imgRGB,imgRGB)),hold=False); title('words assignment')
        scatter(pos[1,:],pos[0,:],10,code)
        subplot(2,1,2)
        bar(range(0,shape(hc)[1]),hc.flatten(),width=1,hold=False); title('image signature')
        draw()

    return signature,Id
    
#    
#     check number of descriptors in a directory
#
def getNumDescriptor(inDir,ext,queue):

    nd = 0 # number of descriptors 
    files,Null = getFilelist(inDir,ext,queue);
    for file in files:
        # print file
        f = open(file, 'rb')
        feats = pickle.load(f)
        f.close()
        s = shape(feats)
        nd = nd + s[0]
            
    return nd


def get_num_hist_pyramid(level):
    """Calculates the number of histograms to be concatenated

    Args:
        level = Levels of the pyramid

    Returns:
        Total number of histograms to be concatenated
    """
    num = 1
    for i in range(2, level+1):
        num += 2**((i-1)*2)
    return num

#
#   compute image regions subdivision for spatial pyramid
#   Week3 - Task 1: implement here the generation of a spatial pyramid of size pyrconf, given an image S   
#
def pyrspaces(S,pyrconf):

    dr = round(S[0]/pyrconf[0])
    dc = round(S[1]/pyrconf[1])
    # each sublist is [r0,r1,c0,c1]
    pyrs = []
    for r in range(pyrconf[0]):
        r0 = r*dr; r1 = r0+dr
    for c in range(pyrconf[1]):
        c0 = c*dc; c1 = c0+dc
    pyrs.append([r0,r1,c0,c1])

    return pyrs


        
#
#     compute occurrency (histogram) of the code in BoW; it includes Spatial Pyramid option
#
def codeHist(code,K,pyr=False,pyrconf=[],S=3,pos=[],img=None,showFig=False):


    #
    #    spatial pyramid
    # Implemented as in the matlab version of the author
    #
    if pyr:
        nh = get_num_hist_pyramid(pyrconf)
        pyrHist = np.ones((1,nh*K))

        h,w, unused =img.shape
        binsHigh = 2**(pyrconf-1)

        pyramid_cell = []
        pyramid_cell.append(np.zeros((binsHigh, binsHigh, K)))

        for i in range (1,binsHigh+1):
            for j in range(1, binsHigh+1):

                #find the coordinates of the current bin
                x_lo = int(np.floor(w/binsHigh * (i-1)))
                x_hi = int(np.floor(w/binsHigh * i))
                y_lo = int(np.floor(h/binsHigh * (j-1)))
                y_hi = int(np.floor(h/binsHigh * j))

                codes_in_path=[]
                counter=0
                for p in pos:
                    if p[0]> x_lo and p[0] <= x_hi and p[1] > y_lo and p[1] <= y_hi:
                        codes_in_path.append(code[counter])
                    counter+=1
                hist = np.zeros((1,K))
                for c in codes_in_path:
                    hist[0, c-1] = hist[0, c-1] + 1
                pyramid_cell[0][i-1,j-1,:] = hist/float(len(code))

        num_bins = binsHigh/2
        for l in range(2, pyrconf+1):
            pyramid_cell.append(np.zeros((num_bins, num_bins, K)))
            for i in range(1, num_bins+1):
                for j in range(1, num_bins+1):
                    pyramid_cell[l-1][i-1,j-1,:] = pyramid_cell[l-2][2*i-2, 2*j-2, :] + pyramid_cell[l-2][2*i-1,2*j-2,:]+ \
                                                   pyramid_cell[l-2][2*i-2, 2*j-1, :] + pyramid_cell[l-2][2*i-1,2*j-1,:]
            num_bins = num_bins/2

        #stack all the histograms with appropriate weights
        pyramid = []
        for l in range(0, pyrconf-1):
            pyr_tmp = pyramid_cell[l].ravel('F')
            pyr = pyr_tmp*2.0**(-l-1)
            pyramid = pyramid + pyr.tolist()
        pyr_tmp = pyramid_cell[pyrconf-1].ravel('F')
        pyr = pyr_tmp*2.0**(1-pyrconf)
        pyramid = pyramid + pyr.tolist()

        return np.array(pyramid)[np.newaxis]


    # 
    # normal histogram
    #
    else:

        hist = zeros((1,K))
        for c in code:
            hist[0,c-1] = hist[0,c-1] + 1
    
        histNorm = hist/sum(hist)    # normalize
        #print sum(hist)
        #print len(code)
        return histNorm    



# plot roc curve in bag of word for svm classifier
def bow_roc(svm_model,x1,x2,name1,name2,nF=20):
    
    nF = 20 # number of folds
    y1 = ones((shape(x1)[0],1))
    y2 = -1*ones((shape(x2)[0],1))
    train_y = list(y1)+list(y2); 
    train_x = map(list,x1)+map(list,x2);
    
    # plot_svm_roc found at mcv_svm_roc.py
    xy_roc,auc = plot_svm_roc(train_x,train_y,svm_model,nFolds=nF)
    xy_roc = array(xy_roc)
                    
    figure(1)
    plot(xy_roc[:,0],xy_roc[:,1])
    xlabel('FP'); ylabel('TP')
    title(''+name1+' vs '+name2+'')
    show()
    
    # area under curve
    return auc


#
#     TRAIN SVM in BAG OF WORDS
#
def train_svm_bow(classifierDir, trainDataDir, bowParams):
    params = bowParams.copy()
    if bowParams['fusion'] == 'late':
        nv = len(bowParams['descriptor'])  # number of vocabularies
    else:
        nv = 1
        
    for voc in range(nv):

        if bowParams['fusion'] == 'early':

            descr = ''
            for d in bowParams['descriptor']: descr = ''+descr+d+'_'
            descr = descr[0:-1]

            nf = get_num_of_features(bowParams)
            '''
            for d in bowParams['descriptor']:
                if d == 'sift': nf = nf + bowParams['nf_sift']
                if d == 'hog': nf = nf + bowParams['nf_hog']
                if d == 'surf': nf = nf + bowParams['nf_surf']
                if d == 'sc': nf = nf + bowParams['nf_sc']
                if d == 'brief': nf = nf + bowParams['nf_brief']
            '''
        else:
            descr = bowParams['descriptor'][voc]
            nf= get_num_of_features_descriptor(descr, bowParams)
            '''
            descr = bowParams['descriptor'][voc]
            if descr == 'sift': nf = bowParams['nf_sift']
            if descr == 'hog': nf = bowParams['nf_hog']
            if descr == 'surf': nf = bowParams['nf_surf']
            if descr == 'sc': nf = bowParams['nf_sc']
            if descr == 'brief': nf = bowParams['nf_brief']
            params = bowParams.copy(); params['descriptor'] = [''+descr+''] # update descriptor for late fusion
            '''
        
        f = open(trainDataDir+descr+'_'+params['keypointDetector']+'_svm_traindata_K_'+str(params['K'])+'.pkl', 'rb')
        trainData = pickle.load(f);
        trainLabels = pickle.load(f)
        f.close()  
                                       
        #
        #     one-vs-all ECOC coding            
        #        
        if bowParams['ecoc-coding'] == 'OneVsAll':
            
            for cidx in range(0,bowParams['nc']):
                
                    #idx1 = find(trainLabels==cidx+1)
                    #idx2 = find(trainLabels!=cidx+1)
                    idx1 = np.where(trainLabels==cidx+1)[0]
                    idx2 = np.where(trainLabels!=cidx+1)[0]
                            
                    x1 = trainData[idx1]
                    x2 = trainData[idx2]
                                                                                 
                    logger_bow.info( 'Learning : '+str(shape(x1)[0])+' positive '+params['classes'][cidx]+' samples and '+str(shape(x2)[0])+' negative samples')
                                               
                    # Week 1 - Task 2: Implement the function trainSVM found at mcv_svm.py        
                    svm_model = trainSVM(x1,x2,params['svm_kernel'])                    
                    svm_save_model(classifierDir+'svm_bow_'+params['keypointDetector']+'_'+descr+'_K_'+str(params['K'])+'_'+params['classes'][cidx]+'_vs_all.model', svm_model)
                
                    ## roc curve (debug)
                    if bowParams['ROC']:
                        AUC = bow_roc(svm_model,x1,x2,params['classes'][cidx],'all')
                    

        
        #
        #     one-vs-one ECOC coding
        #        
        elif bowParams['ecoc-coding'] == 'OneVsOne':

            for cidx1 in range(0,bowParams['nc']-1):
                for cidx2 in range(cidx1+1,bowParams['nc']):

                    logger_bow.info( 'training SVM '+str(cidx1)+' - '+str(cidx2)+'')

                    #idx1 = find(trainLabels==cidx1+1)
                    idx1 = np.where(trainLabels==cidx1+1)[0]
                    #idx2 = find(trainLabels==cidx2+1)
                    idx2 = np.where(trainLabels==cidx2+1)[0]
                        
                    x1 = trainData[idx1,:]
                    x2 = trainData[idx2,:]

                    logger_bow.info( ''+params['classes'][cidx1]+': '+str(shape(x1)[0])+' samples')
                    logger_bow.info( ''+params['classes'][cidx2]+': '+str(shape(x2)[0])+' samples')
                        
                    logger_bow.info( 'TRAIN SVM: Class '+params['classes'][cidx1]+' vs '+params['classes'][cidx2]+'')
                    logger_bow.info( '')

                    # Week 1 - Task 2: Implement the function trainSVM found at mcv_svm.py        
                    svm_model = trainSVM(x1,x2,params['svm_kernel'])                    
                    
                    svm_save_model(classifierDir+'svm_bow_'+params['keypointDetector']+'_'+descr+'_K_'+str(params['K'])+'_'+params['classes'][cidx1]+'_vs_'+params['classes'][cidx2]+'.model', svm_model)
            
                    ## roc curve (debug)
                    if bowParams['ROC']:
                        AUC = bow_roc(svm_model,x1,x2,params['classes'][cidx1],params['classes'][cidx2])

                        
                        

    

