"""

    Bag of Words: TRAINING script
    
    Master in Computer Vision - Barcelona, 2015
    
    Week 1 - Task 1: Compare different vocabulary sizes (bowParams - 'K')   
    Week 1 - Task 2: Compare K-nn vs. svm classifiers (bowParams - 'classifier')
    Week 1 - Task 3: Compare SIFT, Shape Context, SURF and BRIEF descriptors (bowParams - 'descriptor')   
    Week 1 - Task 4: Compress the descriptor, using PCA (bowParams - 'nf_pca')

    Week 2 - Task 1: Use keypoints instead of a dense grid for feature extraction (bowParams - 'keypointDetector')   
    Week 2 - Task 2: Compare single descriptors vs. early fusion (bowParams - 'fusion' and 'descriptor')
    Week 2 - Task 3: Compare global descriptors like GIST and CENTRIST (bowParams - 'descriptor'): 
                     Note that global and local descriptors can not be concatenated using a early fusion estrategy!  
    Week 2 - Task 4: Compare Opponentsift (bowParams - 'descriptor') and color naming technique (colorNaming.py).

    Week 3 - Task 1: Compare different pyramid scheme (bowParams - 'spyramid')   
    Week 3 - Task 2: Compare Early vs. Late Fusion for different settings (bowParams - 'fusion')
    Week 3 - Task 3: Compare LPB (bowParams - 'descriptor')   
    Week 3 - Task 4: Compare color constancy (bowParams - 'cc')

    Authors: Jordi Gonzalez, Ramon Baldrich, Eduard Ramon, Cristhian Aguilera, Miquel Ferrarons

"""

from sys import exit,path
import platform            # system functions for escape
import time
import os
path.append('.')       
import logging
import logging.handlers
 
from mcv_bow import *            # mcv library of bag of words functions
from mcv_svm_roc import *        # mcv library for roc curve
from mcv_svm import *            # mcv module for SVM
import pickle                    # save/load data files1
from argparse2 import *

# ******************  MAIN **************************

if __name__ == "__main__":
        
    #-------------------------------------------------------------------------------------------------------------------
    # Add command line arguments
    #-------------------------------------------------------------------------------------------------------------------
    # Parse arguments

    parser = ArgumentParser(description="Classification train", epilog="")
    parser.add_argument('-ds_folder',       dest='ds_folder',    required=True,
                        help='Datasets folder')
    parser.add_argument('-ds_name',         dest='ds_name',      required=True,
                        help='Dataset folder name')
    parser.add_argument('-out_folder',      dest='out_folder',   required=True,
                        help='Output folder were the results are stored')

    parser.add_argument('-test_folder', '-t_folder', dest='test_folder', required=False, help='Test folder')

    parser.add_argument('-c_descriptor',    dest='c_descriptor', action='store_true',
                        help='Compute descriptor flag')
    parser.add_argument('-c_vocabulary',    dest='c_vocabulary', action='store_true',
                        help='Compute vocabulary flag')
    parser.add_argument('-c_assignment',    dest='c_assignment', action='store_true', 
                        help='Compute assignment flag')

    parser.add_argument('-keypoints',       dest='keypoints',    required=False, default='fast',
                        help='keypoints [dense|sift|surf,...]')
    parser.add_argument('-d',               dest='descriptor',   required=False, nargs='+', default='brief',
                        help='sift|hog|surf|sift hog')
    parser.add_argument('-k',               dest='k',            required=False, type=int, default=100, 
                        help='Number of clusters')
    parser.add_argument('-npt',             dest='npt',          required=False, type=int, default=800,
                        help='keypoints number')
    parser.add_argument('-classifier',      dest='classifier',   required=False, default='knn', 
                        help=' knn|svm')

    parser.add_argument('-t_svm',           dest='t_svm',        action='store_true',
                        help='Train svm flag')
    parser.add_argument('-svm_kernel',      dest='svm_kernel',   required=False, default='-t 2 -g 38 -c 2',
                        help='-t 2 rbf | -t 0 lineal | Read more in svmutils doc')
    parser.add_argument('-m_strategy',      dest='m_strategy',   required=False, default='OneVsOne',
                        help='OveVsAll|OneVsOne')

    parser.add_argument('-pca',             dest='nf_pca',       required=False, type=int, default=0,
                        help='Num of features pca')
    parser.add_argument('-fusion',          dest='fusion',       required=False, default='off',
                        help='off|early|late')
    parser.add_argument('-fusion_param',    dest='fusion_param', required=False, default=0.5,
                        help='value in [0, 1]')
    parser.add_argument('-spyramid',        dest='spyramid',     action='store_true',
                        help='spyramid if it is present')
    parser.add_argument('-pyrconf',         dest='pyrconf',      type=int, default=3,
                        help='levels of the pyramid')
    parser.add_argument('-color_constancy', dest='cc',           default='no',
                        help='Color constancy algorithm: grey_world|retinex|max_white|retinex_adjust')

    parser.add_argument('-debug',           dest='debug',        action='store_true', help='Debug flag')
    parser.add_argument('-parallel',        dest='parallel',     action='store_true', help='Parallel flag')
    parser.add_argument('-roc',             dest='roc',          action='store_true', help='Roc flag')

    #LBP patch size
    parser.add_argument('-lbp_patch_size',  dest='lbp_patch_size', type=int, default=50, help='Size of the lbp patch size')

    print 'Parsing arguments....'
    if len(sys.argv)>1:
        parameters = sys.argv[1]
    else:
        parameters = 'default.parameters'
    text_file = open(parameters )
    args = parser.parse_args(text_file.read().split())
    text_file.close()
    print 'Arguments parsed!'

    #-------------------------------------------------------------------------------------------------------------------
    # End
    #-------------------------------------------------------------------------------------------------------------------

        # define datasets and directories
    DatasetFolder = args.ds_folder
    DataOutputFolder = args.out_folder

    trainDir = DatasetFolder + args.ds_name + '/training'
    valDir = DatasetFolder + args.ds_name+'/validation'
    nc = 5  # number of classes

    descriptorDir = DataOutputFolder + '/descriptors/'
    classifierDir = DataOutputFolder + '/classifiers/'
    trainDataDir  = DataOutputFolder + '/trainData/'

    # TASKS
    #
    # 1. compute image descriptor (SIFT, HOG, SURF ...) in DENSE configuration or in Keypoints    
    computeDescriptors = args.c_descriptor

    #
    # 2. construct words vocabulary through K-MEANS clustering
    computeVocabulary = args.c_vocabulary

    #
    # 3. assign words of vocabulary back to each training image
    computeAssignment = args.c_assignment

    #
    # 4. train SVM classifier on set of words for each class
    train_svm = args.t_svm


    #
    # OPTIONS
    #
    # 1. show figures during TASKS 1-4
    showFig = args.debug

    # 2. Use multi processors
    runmultiprocess= args.parallel
    
    #
    # train files: 5 categories, cars, dogs, bicycles, motorbikes and persons.
    # getFilelist is found at: mcv_tools.py
    #    
    trainFiles_car,filenames_car = getFilelist(''+trainDir+'/car/','jpg') # class 1
    trainFiles_dog,filenames_dog = getFilelist(''+trainDir+'/dog/','jpg') # class 2 
    trainFiles_bicycle,filenames_bike = getFilelist(''+trainDir+'/bicycle/','jpg') # class 3
    trainFiles_motorbike,filenames_moto = getFilelist(''+trainDir+'/motorbike/','jpg') # class 4
    trainFiles_person,filenames_pedestrian = getFilelist(''+trainDir+'/person/','jpg') # class 5
        
    n1 = len(trainFiles_car)
    n2 = len(trainFiles_dog)
    n3 = len(trainFiles_bicycle)
    n4 = len(trainFiles_motorbike)
    n5 = len(trainFiles_person)

    trainFiles = [trainFiles_car]
    trainFiles.append(trainFiles_dog)
    trainFiles.append(trainFiles_bicycle)
    trainFiles.append(trainFiles_motorbike)
    trainFiles.append(trainFiles_person)

    trainLabels = concatenate((zeros(n1)+1, zeros(n2)+2, zeros(n3)+3, zeros(n4)+4, zeros(n5)+5))

            
    # ######################################################################################
    #         BAG OF WORDS - PARAMETERS DEFINITION        
    # ######################################################################################
    #
    # define a dictionary structure to store bag of words parameters 
    #
    # > descriptor: sift, hog, surf; combination of descriptors is allowed, e.g. ['Opponentsift','hog', 'lbp_color'], for fusion
    # > keypointDetector: dense, sift, harris --> 'gridSpacing'/'patchSize' (16) is related with nPt
    # > nf_descriptor_name: number of features
    # > K: number of clusters / size of BOW vocabulary (best value ~ 200)
    # > nPt: numer of keypoints to selecte from each training image, related to gridSpacing
    # > classifier: knn, svm    
    # > fusion: off, early, late
    # > classes: list of class names
    # > ecoc-coding: OneVsAll
    # > spyramid: spatial pyramid; pyrconf: number of levels of the spatial pyramid
    bowParams = {'patchSize': 10, 'gridSpacing': 4, 'keypointDetector': args.keypoints,
                   'harris_sigma':3, 'harris_min_dist':1, 'harris_threshold':0.01, 'harris_wid':5,\
                   'descriptor': args.descriptor,
                   'K': args.k, 'nPt': args.npt,
                   'nf_gist':60,\
                   'nf_sift':128, 'sift_edge-thresh':10, 'sift_peak-thresh':5,\
                   'nf_surf':128, 'surf_extended':1,'surf_hessianThreshold':5.0, 'surf_nOctaveLayers':1, 'surf_nOctaves':1,\
                   'nf_hog':81,\
                   'nf_brief':32,\
                   'nf_brisk': 64, \
                   'nf_freak': 64, \
                   'nf_gist': 60, \
                   'nf_orb': 32, \
                   'nf_centrist': 10, \
                   'nf_sc': 60, \
                   'nf_lbp': 59, \
                   'nf_lbp_color':177, \
                   'lbp_patch_size': int(args.lbp_patch_size), \
                   'classifier': args.classifier, \
                   'svm_kernel': args.svm_kernel, \
                   'fusion': args.fusion, \
                   'fusion_param': args.fusion_param, \
                   'nc': nc,\
                   'nf_pca': args.nf_pca, \
                   'ROC': args.roc, \
                   'cc': args.cc, \
                   'classes':['car','dog','bicycle','motorbike','person'], \
                   'ecoc-coding': args.m_strategy, \
                   'spyramid': args.spyramid, \
                   'pyrconf': args.pyrconf}

    if len(bowParams['descriptor'])>1 and bowParams['fusion'] == 'off':
        print 'ERROR: DESCRIPTORS incompatible with FUSION type!'
        exit(-1)
        
    if bowParams['descriptor'][0] == 'surf' and bowParams['keypointDetector']!= 'surf':
        print 'ERROR: SURF descriptor is only compatible with SURF keypoints!'
        exit(-1)
    

    # ###############################
    #             TRAINING
    # Functions below are located at: mcv_bow.py
    # ###############################    
    
    #    
    # 1. compute descriptors of 'nPt' points per image for vocabulary    
    #    
    if not os.path.isdir(DataOutputFolder): os.mkdir(DataOutputFolder)
    if not os.path.isdir(descriptorDir): os.mkdir(descriptorDir)
    if not os.path.isdir(classifierDir): os.mkdir(classifierDir)
    if not os.path.isdir(trainDataDir):  os.mkdir(trainDataDir)
     
    #-------------------------------------------------------------------------------------------------------------------
    # Create log to store results
    #-------------------------------------------------------------------------------------------------------------------
    # Create logger
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(DataOutputFolder+'log', mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('//========================================================//')
    logger.info('Training starts')
    logger.info('//========================================================//')
    logger.info('keypointDetector : (%s) ' % (bowParams['keypointDetector']))
    logger.info('Descriptor(s) : (%s) ' % (bowParams['descriptor']))
    logger.info('Vocabulary size : (%d) ' % (bowParams['K']))
    logger.info('Number of features : (%d) ' % (bowParams['nPt']))
    logger.info('Fusion : (%s) ' % (bowParams['fusion']))
    logger.info('Spyramid : (%s) ' % (bowParams['spyramid']))
    logger.info('runmultiprocess : (%s) ' % runmultiprocess)
    
    if bowParams['classifier'] == 'knn':
        logger.info('Classifier : (%s) ' % (bowParams['classifier']))
    else:
        logger.info('Classifier : (%s) ' % (bowParams['classifier']))
        logger.info('SVM Kernel : (%s) ' % (bowParams['svm_kernel']))

    #-------------------------------------------------------------------------------------------------------------------
    # End
    #-------------------------------------------------------------------------------------------------------------------

   #
    # 1. compute descriptors of 'nPt' points per image for vocabulary
    #
    logger.info('Step 1/4: Computing descriptors...')
    ts = time.time()
    if computeDescriptors:
        getDescriptor(trainFiles, descriptorDir, bowParams, showFig, runmultiprocess)
    te = time.time()
    logger.info('Step 1/4 finished in %2.2f sec' % (te-ts))

    #
    # 2. compute vocabulary by clustering the descriptors: 
    #
    logger.info('Step 2/4: Computing vocabulary...')
    ts = time.time()
    if computeVocabulary:
        getVocabulary(descriptorDir, 'pkl', classifierDir, bowParams)
    te = time.time()
    logger.info('Step 2/4 finished in %2.2f sec' % (te-ts))

    #
    # 3. words assignment to images
    #
    logger.info('Step 3/4: Computing assignment...')
    ts = time.time()
    if computeAssignment:
        getAssignment(trainFiles, trainLabels, classifierDir, trainDataDir, bowParams, showFig, runmultiprocess)
    te = time.time()
    logger.info('Step 3/4 finished in %2.2f sec' % (te-ts))

    #
    # 4. train SVM
    #
    logger.info('Step 4/4: Computing classifiers...')
    if bowParams['classifier'] == 'svm' and train_svm:
        logger.info('Training svm')
        ts = time.time()
        train_svm_bow(classifierDir, trainDataDir, bowParams)
        te = time.time()
        logger.info('Step 4/4 finished in %2.2f sec' % (te-ts))

   
    logger.info('//==========================================================//')
    logger.info('Training done')
    logger.info('//==========================================================//')
    logger.info(' ')        
    
    



