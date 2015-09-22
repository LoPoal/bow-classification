""" 
    
    Plot ROC curve for SVM
    
    Master in Computer Vision - Barcelona
    
    Author: Francesco Ciompi, Ramon Baldrich, Jordi Gonzalez
         
"""

from random import randrange , seed
from svmutil import *
from operator import itemgetter
import platform
if not platform.node().lower().startswith('compute'):
    from pylab import * # matplotlib, for graphical plots

"""

    partially based on plotroc.py (http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#roc_curve_for_binary_svm)

"""


#get_cv_deci(prob_y[], prob_x[], svm_parameter param, nr_fold)
#input raw attributes, labels, param, cv_fold in decision value building
#output list of decision value, remember to seed(0)
def get_cv_deci(prob_y, prob_x, param, nr_fold):
    if nr_fold == 1 or nr_fold==0:
        deci,model = get_pos_deci(prob_y, prob_x, prob_y, prob_x, param)
        return deci
    deci, model = [], []
    prob_l = len(prob_y)

    #random permutation by swapping i and j instance
    for i in range(prob_l):
        j = randrange(i,prob_l)
        prob_x[i], prob_x[j] = prob_x[j], prob_x[i]
        prob_y[i], prob_y[j] = prob_y[j], prob_y[i]

    #cross training : folding
    for i in range(nr_fold):
        begin = i * prob_l // nr_fold
        end = (i + 1) * prob_l // nr_fold
        train_x = prob_x[:begin] + prob_x[end:]
        train_y = prob_y[:begin] + prob_y[end:]
        test_x = prob_x[begin:end]
        test_y = prob_y[begin:end]
        subdeci, submdel = get_pos_deci(train_y, train_x, test_y, test_x, param)
        deci += subdeci
    return deci


def get_pos_deci(test_y, test_x, model, train_x, train_y, param):
        
    # roc on training    
    if param != None:
        model = svm_train(train_y, train_x, param)
    
    #predict and grab decision value, assure deci>0 for label+,
    #the positive descision value = val[0]*labels[0]
    labels = model.get_labels()
    py, evals, deci = svm_predict(test_y, test_x, model)
    deci = [labels[0]*val[0] for val in deci]
    return deci,model


def plot_roc(deci, label):
    #count of postive and negative labels
    db = []
    pos, neg = 0, 0         
    for i in range(len(label)):
        if label[i]>0:
            pos+=1
        else:    
            neg+=1
        db.append([deci[i], label[i]])

    #sorting by decision value
    db = sorted(db, key=itemgetter(0), reverse=True)

    #calculate ROC 
    xy_arr = []
    tp, fp = 0., 0.            #assure float division
    for i in range(len(db)):
        if db[i][1]>0:        #positive
            tp+=1
        else:
            fp+=1
        xy_arr.append([fp/neg,tp/pos])

    
    #area under curve
    auc = 0.            
    prev_x = 0
    for x,y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x

    # x: false positive (column 0)
    # y: true positive (column 1)
    xy_roc = xy_arr
    return xy_roc,auc



def plot_svm_roc(test_x, test_y, model, train_x=None,train_y=None,params=None,nFolds=None):
    
    # consider the case: roc over training (with cross-fold) and roc over test (with two datasets)

    # for the roc over test, we should use
    # > deci,model = get_pos_deci(train_y, train_x, test_y, test_x, param)
    # and passing the model already trained
    
    # roc on test
    if train_x == None:
        
        labels = model.get_labels()
        py, evals, deci = svm_predict(test_y, test_x, model)
        deci = [labels[0]*val[0] for val in deci]
        
        xy_roc,auc = plot_roc(deci, test_y)        
    
    # roc on training
    else:
    
        deci = get_cv_deci(train_y, train_x,'-t 0', nFolds)
        xy_roc,auc = plot_roc(deci, train_y)
    
    return xy_roc,auc
    
    
    
    
    
    
    
    
    
    
