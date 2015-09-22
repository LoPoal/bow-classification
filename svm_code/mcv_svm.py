from imtools import *	# personal library of tools for image processing
from PIL import Image 	# python image library
from numpy import *	# numpy, for maths computations
from svmutil import *

# train SVM
def trainSVM(x1,x2,params):
	
	x1 = map(list,x1)
	x2 = map(list,x2)
	
	X = x1+x2
	y1 = ones((shape(x1)[0],1))
	y2 = -1*ones((shape(x2)[0],1))
	Y = list(y1)+list(y2)
	
	prob = svm_problem(Y,X)
	param = svm_parameter('-t 2')
	m = svm_train(prob,param)
	
	return m


# test SVM
def testSVM(x,y_gt):

	pass
	

# ******************  MAIN **************************
if __name__ == "__main__":
	
	# Training data
	c1TrainDir1 = '../datasets/DatasetClassification/train/background/'
	c2TrainDir1 = '../datasets/DatasetClassification/train/circles/'
	c2TrainDir2 = '../datasets/DatasetClassification/train/giveways/'
	c2TrainDir3 = '../datasets/DatasetClassification/train/rectangles/'
	c2TrainDir4 = '../datasets/DatasetClassification/train/stops/'
	c2TrainDir5 = '../datasets/DatasetClassification/train/triangles/'
	
	# Test data
	c1TestDir1 = '../datasets/DatasetClassification/test/background/'
	c2TestDir1 = '../datasets/DatasetClassification/test/circles/'
	c2TestDir2 = '../datasets/DatasetClassification/test/giveways/'
	c2TestDir3 = '../datasets/DatasetClassification/test/rectangles/'
	c2TestDir4 = '../datasets/DatasetClassification/test/stops/'
	c2TestDir5 = '../datasets/DatasetClassification/test/triangles/'
	
	files_c1_1,Null = getImlist(c1TrainDir1,'jpg');
	files_c2_1,Null = getImlist(c2TrainDir1,'ppm');
	files_c2_2,Null = getImlist(c2TrainDir2,'ppm');
	files_c2_3,Null = getImlist(c2TrainDir3,'ppm');
	files_c2_4,Null = getImlist(c2TrainDir4,'ppm');
	files_c2_5,Null = getImlist(c2TrainDir5,'ppm');
	
	TRAIN = False
	TEST = True
	
	if TRAIN:
		x1 = featureMatrix(files_c1_1,'HOG')
		x2_1 = featureMatrix(files_c2_1,'HOG')
		x2_2 = featureMatrix(files_c2_2,'HOG')
		x2_3 = featureMatrix(files_c2_3,'HOG')
		x2_4 = featureMatrix(files_c2_4,'HOG')
		x2_5 = featureMatrix(files_c2_5,'HOG')
	
		x2 = vstack((x2_1,x2_2,x2_3,x2_4,x2_5))
		
		# --> save features <--
		
		# TRAIN SVM
		# -t 'kernel'
		# 0 linear, 1 polynomial, 2 RBF (default), 3 sigmoid
		svm_params = '-t 2'
		svm_model = trainSVM(x1,x2,svm_params)
	
		svm_save_model('svm_traffic_sign.model', svm_model)
	
	# TEST SVM
	test_files_c1_1,Null = getImlist(c1TestDir1,'jpg');
	test_files_c2_1,Null = getImlist(c2TestDir1,'ppm');
	test_files_c2_2,Null = getImlist(c2TestDir2,'ppm');
	test_files_c2_3,Null = getImlist(c2TestDir3,'ppm');
	test_files_c2_4,Null = getImlist(c2TestDir4,'ppm');
	test_files_c2_5,Null = getImlist(c2TestDir5,'ppm');
	
	if TEST:
		x_test_1 = map(list,featureMatrix(test_files_c1_1,'HOG'))
		x_test_2 = map(list,featureMatrix(test_files_c2_1,'HOG'))
		x_test_3 = map(list,featureMatrix(test_files_c2_2,'HOG'))
		x_test_4 = map(list,featureMatrix(test_files_c2_3,'HOG'))
		x_test_5 = map(list,featureMatrix(test_files_c2_4,'HOG'))
		x_test_6 = map(list,featureMatrix(test_files_c2_5,'HOG'))
		
		# --> save features <--
		
		x_test = map(list,vstack((x_test_1,x_test_2,x_test_3,x_test_4,x_test_5,x_test_6)))
	
		y1 = list(ones((shape(x_test_1)[0],1)))
		y2 = list(-1*ones((shape(x_test_2)[0],1)))
		y3 = list(-1*ones((shape(x_test_3)[0],1)))
		y4 = list(-1*ones((shape(x_test_4)[0],1)))
		y5 = list(-1*ones((shape(x_test_5)[0],1)))
		y6 = list(-1*ones((shape(x_test_6)[0],1)))
		
		y_gt = y1+y2+y3+y4+y5+y6
		
		svm_model = svm_load_model('libsvm.model')
		
		y_auto = svm_predict(y_gt,x_test,svm_model)
	
	print y_auto











