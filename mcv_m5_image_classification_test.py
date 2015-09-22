"""

    Bag of Words: TESTING script
    
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

from sys import exit,path            # system functions for escape
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn import metrics

path.append('.')        
from mcv_bow import *
import mcv_ecoc as ecoc            # mcv library for error-correcting output codes
from sys import platform
import matplotlib
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import *       # matplotlib, for graphical plots
from argparse2 import *



def test_image(filename, bowParams, num_im, class_real, signatures, pca):

    imgRGB = array(Image.open(filename))
    img = cv2.imread(filename)
    img = apply_color_constancy(img, bowParams)
        
    if bowParams['fusion'] == 'late':
        code_late = []

    for voc in range(nv):

        # update descriptor field for late fusion
        if bowParams['fusion'] == 'late' or bowParams['fusion'] == 'off':
            descr = bowParams['descriptor'][voc]
            params = bowParams.copy(); params['descriptor'] = [''+descr+'']
        else:
            descr = '';
            for d in bowParams['descriptor']: descr = ''+descr+d+'_'
            descr = descr[0:-1]   
            params = bowParams.copy(); 

        # load centroids for the corresponding classifier
        f = open(bowParams['classifierDir']+bowParams['keypointDetector']+'_'+descr+'_centroids.pkl', 'rb')
        centroids = pickle.load(f)
        meanX = pickle.load(f)
        stdX = pickle.load(f)
        f.close()             

        if bowParams['classifier'] == 'knn':           
            f = open(trainDataDir+descr+'_'+params['keypointDetector']+'_bow_signatures_K_'+str(params['K'])+'.pkl', 'rb')
            signatures = pickle.load(f)
            f.close()            

        # compute and normalize features
        feats,pos = getFeatImg(img,params,num_im,filename)
            
        feats = normalizeFeat(feats,meanX,stdX)[0]
            
        #MIQUEL PCA
        if params['nf_pca'] > 0:
            feats = pca.transform(feats)
        #END PCA

        # compute histogram of words in the image
        codes = vq(feats,centroids)[0]
        
        if params['spyramid'] and params['descriptor'] != ['gist']:
            S = shape(img)
            h = codeHist(codes,params['K'],bowParams['spyramid'],bowParams['pyrconf'],S,pos,imgRGB,showFig)
        else:
            h = codeHist(codes,params['K'])                                

        # test knn
        if bowParams['classifier'] == 'knn' and bowParams['fusion'] != 'late':                
            y_auto = yMinDist(h,signatures,'euclidean')+1
                
        if bowParams['classifier'] == 'knn' and bowParams['fusion'] == 'late':                      
            y_auto = yMinDist(h,signatures,'euclidean')+1
            code_late.append(y_auto)
            
        # test svm
        if bowParams['classifier'] == 'svm':

            # initialize ECOC matrix for this multi-class svm                    
            # the modules for ecoc can be found in mcv_ecoc.py
            M = ecoc.matrix(nc,params['ecoc-coding'])
                
            code = zeros((1,shape(M)[1]))
            decis = zeros((1,shape(M)[1]))   #<--- versio poal
            
            # one-vs-all    
            if bowParams['ecoc-coding'] == 'OneVsAll':
                for cidx in range(0,nc):
                    svm_model = svm_load_model(classifierDir+'svm_bow_'+bowParams['keypointDetector']+'_'+descr+'_K_'+str(params['K'])+'_'+params['classes'][cidx]+'_vs_all.model')                                                       
                    #y_bin, evals, deci = svm_predict(list([-1]),map(list,h),svm_model)                        
            
                    decis =  svm_model.predict_proba(h)   

                    if svm_model.classes_[0] == 1:
                        code[0][cidx] = decis[0][0] 
                    else:
                        code[0][cidx] = decis[0][1] 
                                                            
                voting = argmax(code[0])+1 
                                                 
                                       
            # one-vs-one    
            elif bowParams['ecoc-coding'] == 'OneVsOne':
                for cidx in range(0,nc):
                    svm_model = svm_load_model(classifierDir+'svm_bow_'+bowParams['keypointDetector']+'_'+descr+'_K_'+str(params['K'])+'_'+params['classes'][cidx]+'_vs_all.model')                                                       

                    decis =  svm_model.predict_proba(h)   

                    if svm_model.classes_[0] == 1:
                        code[0][cidx] = decis[0][0] 
                    else:
                        code[0][cidx] = decis[0][1] 
                
                voting = code.copy()
            
            if bowParams['fusion'] == 'late':
                #code_late.append(code)
                code_late.append(voting)
            else:
                # the modules for ecoc can be found in mcv_ecoc.py
            
                y_auto = int(voting) #voting.argmax()+1       #<--- versio poal
                
    # decoding for late fusion
    if bowParams['fusion'] == 'late':
        if bowParams['classifier'] == 'svm':
            M_late = tile(M,(1,nv))

            # the modules for ecoc can be found in mcv_ecoc.py
            if bowParams['ecoc-coding'] != 'OneVsAll':
                code_late = reshape(array(code_late).flatten(),(1,nv*nc))
                y_auto = ecoc.decode(code_late,M_late)[0][0]
            else:
                y_auto = max(set(code_late), key=code_late.count)            
        else:
            y_auto = max(set(code_late), key=code_late.count)            
            
    return (h, class_real, y_auto, num_im) 
    
# ******************  MAIN **************************
if __name__ == "__main__":
    
    #-------------------------------------------------------------------------------------------------------------------
    # New code: Add command line arguments
    #-------------------------------------------------------------------------------------------------------------------
    # Parse arguments

    #parser = argparse.ArgumentParser(description="Classification train", epilog="Eduard,Cristhian,Miquel")
    parser = ArgumentParser(description="Classification train", epilog="Eduard,Cristhian,Miquel")
    parser.add_argument('-ds_folder', dest='ds_folder', required=True, help='Datasets folder')
    parser.add_argument('-ds_name', dest='ds_name', required=True, help='Dataset folder name')
    parser.add_argument('-out_folder', dest='out_folder', required=True, help='Output folder were the results '
                                                                              'are stored')
    parser.add_argument('-test_folder', '-t_folder', dest='test_folder', required=False, help='Test folder')

    parser.add_argument('-c_descriptor', dest='c_descriptor', action='store_true', help='Computer descriptor flag')
    parser.add_argument('-c_vocabulary', dest='c_vocabulary', action='store_true', help='Computer vocabulary flag')
    parser.add_argument('-c_assignment', dest='c_assignment', action='store_true', help='Computer assignment flag')
    parser.add_argument('-t_svm', dest='t_svm', action='store_true', help='Train svm flag')

    parser.add_argument('-svm_kernel', '-s_kernel', dest='svm_kernel', required=False, default='-t 2 -g 38 -c 2',
                        help='-t 2 rbf | -t 0 lineal | Read more in svmutils doc')
    parser.add_argument('-m_strategy', dest='m_strategy', required=False, default='OneVsOne',
                        help='OveVsAll|OneVsOne')

    parser.add_argument('-keypoints', dest='keypoints', required=False, default='fast',
                        help='keypoints [dense|sift|surf,...]')
    parser.add_argument('-d', dest='descriptor', required=False, nargs='+', default='brief',
                        help='sift|hog|surf|sift hog')
    parser.add_argument('-k', dest='k', required=False, type=int, default=100, help='Number of clusters')
    parser.add_argument('-npt', dest='npt', required=False, type=int, default=800, help='keypoints number')
    parser.add_argument('-classifier', dest='classifier', required=False, default='knn', help=' knn|svm')
    parser.add_argument('-pca', dest='nf_pca', type=int, required=False, default=0, help='Num of features pca')
    parser.add_argument('-fusion', dest='fusion', required=False, default='off', help='off|early|late')
    parser.add_argument('-fusion_param', dest='fusion_param', type=float, required=False, default=0.5, help='value in [0, 1]')
    parser.add_argument('-spyramid', dest='spyramid',  action='store_true', help='spyramid if it is present')
    parser.add_argument('-pyrconf', dest='pyrconf', type=int, default=3, help='levels of the pyramid')
    parser.add_argument('-color_constancy', '-cc', dest='cc',  default='no', help='Color constancy algorithm'
                                                                                  ' grey_world|retinex|max_white|'
                                                                                  'retinex_adjust')

    parser.add_argument('-debug', dest='debug', action='store_true', help='Debug flag')
    parser.add_argument('-parallel',        dest='parallel',     action='store_true', help='Parallel flag')
    parser.add_argument('-roc', dest='roc', action='store_true', help='Roc flag')

    #LBP patch size
    parser.add_argument('-lbp_patch_size', dest='lbp_patch_size', type=int, default=50, help='Size of the lbp patch size')

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

    # define datasets and directories (using the args)
    DatasetFolder = args.ds_folder
    DataOutputFolder = args.out_folder

    trainDir = DatasetFolder + args.ds_name + '/train'
    valDir = DatasetFolder + args.ds_name+'/validation'
    testDir = DatasetFolder + args.ds_name+'/'+args.test_folder
    nc = 5 # number of classes

    #descriptorDir = './descriptors/';
    descriptorDir = DataOutputFolder + 'descriptors/'
    classifierDir = DataOutputFolder + 'classifiers/'
    trainDataDir  = DataOutputFolder + 'trainData/'
    testDataDir   = DataOutputFolder + 'testing/'

    if not os.path.isdir(testDataDir):  os.mkdir(testDataDir)
    
    #
    # test files: 'car','dog','bicycle','motorbike','person'
    #    
    testFiles_car,filenames_car = getFilelist(''+testDir+'/car/','jpg'); # class 1
    testFiles_dog,filenames_dog = getFilelist(''+testDir+'/dog/','jpg'); # class 2 
    testFiles_bicycle,filenames_bike = getFilelist(''+testDir+'/bicycle/','jpg'); # class 3
    testFiles_motorbike,filenames_moto = getFilelist(''+testDir+'/motorbike/','jpg'); # class 4
    testFiles_person,filenames_pedestrian = getFilelist(''+testDir+'/person/','jpg'); # class 5
            
    testFiles = [];
    testFiles.append(testFiles_car); n1 = len(testFiles_car)
    testFiles.append(testFiles_dog); n2 = len(testFiles_dog)
    testFiles.append(testFiles_bicycle); n3 = len(testFiles_bicycle)
    testFiles.append(testFiles_motorbike); n4 = len(testFiles_motorbike)
    testFiles.append(testFiles_person); n5 = len(testFiles_person)

    #ntest = [n1, n2, n3, n4, n5]
    #
    # OPTIONS
    #
    # 1. show figures
    showFig = args.debug
    if platform.node().lower().startswith('compute'):
        showFig = False

    #
    # 2. compute Receive Operator Curve (ROC) for both train and test
    ROC = args.roc
    
    # 2. Use multi processors
    runmultiprocess= args.parallel

    # ######################################################################################
    #         BAG OF WORDS - PARAMETERS DEFINITION        
    # ######################################################################################
    #
    #     define a dictionary structure to store bag of words parameters 
    #
    # > descriptor: sift, hog, surf; combination of descriptors is allowed, e.g. ['sift','hog'], for fusion
    # > keypointDetector: dense, sift, harris
    # > nf_descriptor_name: number of features
    # > K: number of clusters / size of BOW vocabulary (best value ~ 200)
    # > nPt: numer of keypoints to selecte from each training image
    # > classifier: knn, svm    
    # > fusion: off, early, late
    # > classes: list of class names
    # > ecoc-coding: OneVsAll
    bowParams = {    'patchSize':10, 'gridSpacing':4, 'keypointDetector': args.keypoints,\
                       'harris_sigma':3, 'harris_min_dist':1, 'harris_threshold':0.01, 'harris_wid':5,\
                       'descriptor':args.descriptor,\
                       'K': args.k, 'nPt': args.npt,\
                       'nf_gist':60,\
                       'nf_sift':128, 'sift_edge-thresh':10, 'sift_peak-thresh':5,\
                       'nf_surf':128, 'surf_extended':1,'surf_hessianThreshold':5.0, 'surf_nOctaveLayers':1, 'surf_nOctaves':1,\
                       'nf_hog':81,\
                       'nf_brief':32,\
                       'nf_brisk': 64,
                       'nf_freak': 64,
                       'nf_gist': 60,
                       'nf_orb': 32,
                       'nf_centrist': 10,
                       'nf_sc': 60,
                       'nf_lbp': 59,  #uniform
                       'nf_lbp_color':177,
                       'lbp_patch_size': args.lbp_patch_size,
                       'nf_centrist': 10,\
                       'classifier': args.classifier,
                       'fusion': args.fusion,
                       'fusion_param': args.fusion_param,
                       'nc': nc,\
                       'nf_pca': args.nf_pca, \
                       'ROC': args.roc,
                       'cc': args.cc,\
                       'classes':['car','dog','bicycle','motorbike','person'],\
                       'spyramid': args.spyramid,
                       'pyrconf' :args.pyrconf,
                       'ecoc-coding': args.m_strategy,
                       'svm_kernel': args.svm_kernel,
}
                       

    if len(bowParams['descriptor'])>1 and bowParams['fusion'] == 'off':
        print 'ERROR: DESCRIPTORS incompatible with FUSION type!'
        exit(-1)
        
    if bowParams['descriptor'][0] == 'surf' and bowParams['keypointDetector']!= 'surf':
        print 'ERROR: SURF descriptor is only compatible with SURF keypoints!'
        exit(-1)

    #-------------------------------------------------------------------------------------------------------------------
    # New code: Create log to store results
    #-------------------------------------------------------------------------------------------------------------------
    # Create logger
    logger = logging.getLogger('logTest')
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(DataOutputFolder+'/logtest', mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('//===============================================================')
    logger.info('Test starts')
    logger.info('//===============================================================')
    logger.info('keypointDetector : (%s) ' % (bowParams['keypointDetector']))
    logger.info('Descriptor : (%s) ' % (bowParams['descriptor']))
    logger.info('K : (%d) ' % (bowParams['K']))
    logger.info('Number of features : (%d) ' % (bowParams['nPt']))
    logger.info('Fusion: (%s) ' % (bowParams['fusion']))
    logger.info('Spyramid: (%s) ' % (bowParams['spyramid']))
    if bowParams['classifier'] == 'knn':
        logger.info('Classifier : (%s) ' % (bowParams['classifier']))
    else:
        logger.info('Classifier : (%s) ' % (bowParams['classifier']))
        logger.info('Ecoc-coding : (%s) ' % (bowParams['ecoc-coding']))
        logger.info('SVM Kernel : (%s) ' % (bowParams['svm_kernel']))
    logger.info('Test folder : (%s) ' % testDir)
    #Results log
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh2 = logging.FileHandler(DataOutputFolder+'/results')
    fh2.setLevel(logging.INFO)
    logger_results.addHandler(fh2)
    #-------------------------------------------------------------------------------------------------------------------
    # End
    #-------------------------------------------------------------------------------------------------------------------
    ts = time.time()

    # confusion matrix
    confMat = zeros((nc,nc))
        
    # compute number of rounds per test image    
    if bowParams['fusion'] == 'late':
        nv = len(bowParams['descriptor']) # number of vocabularies
    else:
        nv = 1
        
    # working copy of bowParams
    bowParams['classifierDir'] = classifierDir       
    params = bowParams.copy();

    if bowParams['fusion'] == 'early' or bowParams['fusion'] == 'off':
        descr = '';
        for d in bowParams['descriptor']: descr = ''+descr+d+'_'
        descr = descr[0:-1]
        
    if bowParams['classifier'] == 'knn' and bowParams['fusion'] == 'early':     #### change late knn
        f = open(trainDataDir+descr+'_'+params['keypointDetector']+'_bow_signatures_K_'+str(params['K'])+'.pkl', 'rb')
        signatures = pickle.load(f)
        f.close()
    else: signatures = None

    if bowParams['spyramid'] and bowParams['descriptor'] != ['gist']:
        nh = get_num_hist_pyramid(bowParams['pyrconf']) # number of histograms to join

    # scan groups of files per class
    y_test = 0
    #print len(testFiles[1])
    
    ncontconf = 0    
    
    for files in testFiles:
            
        y_test+=1
        
        pca = []
        #MIQUEL PCA
        nf_pca= params['nf_pca']
        if nf_pca > 0:
            filepath= classifierDir+bowParams['keypointDetector']+'_PCA'+str(nf_pca)+'.pkl'
            logger.info( "LOADING PCA FROM " + filepath)
            f = open(filepath, 'rb')
            pca = pickle.load(f)
        #END MIQUEL PCA
                    
        test_class_x = zeros((len(files),params['K']))
        test_class_y = ones((len(files),1))*y_test
        
        if ROC:    # initialize test descriptor matrix for roc curve in binary problem
            if bowParams['spyramid'] and bowParams['descriptor'] != ['gist']:
                test_class_x = zeros((len(files),nh*params['K']))
            else:
                test_class_x = zeros((len(files),params['K']))
            #print shape(test_class_x)
            test_class_y = ones((len(files),1))
            classCont = 0            
                    
        # scan all files of one class
        if runmultiprocess: 
            pool = mp.Pool(mp.cpu_count()- 1 if mp.cpu_count()>1 else 0)
            PoolResults = []
                    
        num_im = 0
        for testFile in files:
        
            #logger.info(testFile)

            if runmultiprocess:         ########### start change
                PoolResults.append(pool.apply_async(test_image, args=(testFile,bowParams, num_im, y_test, signatures, pca)))
            else:
                (h, class_real, class_assigned, num_im) = test_image(testFile,bowParams, num_im, y_test, signatures, pca)    
                 
                if ROC:    # store descriptor
                    test_class_x[num_im] = h
                    
                ncontconf += 1
                
                print class_real, class_assigned, ncontconf
                
                confMat[class_real-1,class_assigned-1] += 1
    
                # show result of one test
                if showFig:
                    ion()
                    figure('test')
                    subplot(2,1,1); imshow(hstack((imgRGB,imgRGB)),hold=False)                    
                    title('auto label: '+bowParams['classes'][int(y_auto)-1]+'')
                    scatter(pos[1,:],pos[0,:],20,codes)   
                    subplot(2,1,2); imshow(confMat,hold=False)
                    title('confusion matrix')
                    draw()
            num_im += 1

        
        if runmultiprocess:
            pool.close()
            while (len(PoolResults)>0):
                try:
                    h, class_real, class_assigned, num_im = PoolResults[0].get(timeout=0.001)
                    PoolResults.pop(0)
                    if ROC:    # store descriptor
                        test_class_x[num_im] = h
                        
                    confMat[class_real-1,class_assigned-1] += 1
                    test_class_x[num_im] = h
                    ncontconf += 1
                    
                    #print num_im, ': ', class_real, class_assigned, ncontconf, np.sum(confMat)    
                    logger.info('True class : '+str(class_real)+' vs. Predicted class: '+str(class_assigned))  
                except:
                    pass
            pool.join()

        #if ROC:
        # save test data for this class, to be used in the ROC evaluation
        f = open(testDataDir +'test_data_class_'+str(y_test)+'.pkl', 'wb')
        pickle.dump(test_class_x,f);
        pickle.dump(test_class_y,f);
        f.close()
#    
    logger.info('Confusion Matrix:')
    logger.info(confMat) 
    ntest = (np.sum(confMat))

    logger.info('Accuracy = %2.2f of %2.0f elements' % (accuracy(confMat), ntest))
    
       
    logger.info('//==========================================================//')
    logger.info('Testing done')
    logger.info('//==========================================================//')
    logger.info(' ')        






