""" 
    Training SVM 
    
    Master in Computer Vision - Barcelona
    
    Week 1 - Task 2: Implement the function trainSVM found       

    Author: Francesco Ciompi, Ramon Baldrich, Jordi Gonzalez
         
"""


from svmutil import *
from numpy import *    # numpy, for maths computations

from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import recall_score
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer


# Week 1 - Task 2: train SVM with diffrerent kernels
def trainSVM(x1,x2,kernel):
    # prepare data  
    x1 = map(list,x1)
    x2 = map(list,x2)
           
    X = x1+x2
    y1 = ones((shape(x1)[0],1))
    y2 = -1*ones((shape(x2)[0],1))
    Y = list(y1)+list(y2)
    Y = ravel(Y)
    #print 'Y'   
    if (kernel == 0):
        svm = LinearSVC()                               #Instantiating the SVM LINEAR classifier.
        params = {'C': [1, 10, 50, 100,200,300]}                    #Defining the params C which will be used by GridSearch. Param C does increase the weight of the 'fails'.
        grid = GridSearchCV(svm, params, cv=5)
    else:
        svm = SVC(probability=True)                                     #Instantiating the SVM RBF classifier.
        params = {'C': [50, 100,200,300]} #Defining the params C & Gamma which will be used by GridSearch. Param C does increase the weight of the 'fails'. Gamma does define the std of a gaussian.
        grid = GridSearchCV(svm, params, cv=5)
        
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)   
    trainData = imp.transform(X)

    grid.fit(trainData, Y)        #Run fit with all sets of parameters.
    model = grid.best_estimator_
    return model

#def svm_save_model(model_file_name, model):
#
#    """
#    svm_save_model(model_file_name, model) -> None
#
#    Save a LIBSVM model to the file model_file_name.
#    """
#    libsvm.svm_save_model(model_file_name.encode(), model)
    
    
def svm_save_model(model_file_name, model):
	joblib.dump(model,model_file_name)

def svm_load_model(model_file_name):
        model = joblib.load(model_file_name)
        return model
        