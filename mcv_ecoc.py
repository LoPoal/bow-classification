""" 
	
	Error-Correcting Output Codes (ECOC)
	
	Master in Computer Vision - Barcelona
	
	Author: Francesco Ciompi
		 
"""

from numpy import *

#
# 	ECOC matrix
#
def matrix(nc,coding):
	
	if nc>1:
		if coding == 'OneVsOne': # one-versus-one design
			ECOC = zeros((nc,nc*(nc-1)/2),float);
			counter = 0;
			for i in range(0,nc-1):				
				for j in range(i+1,nc):
					ECOC[i,counter] = 1;
					ECOC[j,counter] = -1;
					counter = counter + 1;
					
		if coding == 'OneVsAll': # one-versus-all design
			ECOC = -1*ones((nc,nc));
			for i in range(0,nc):
				ECOC[i,i] = 1;

	return ECOC
	
	
#	
# 	Decode ECOC
#
def decode(codes,ECOC):
	
	# euclidean distance
	if shape(codes)[1] != shape(ECOC)[1]:
		print "ECOC matrix and codes have different number of columns!"
		pass
	
	y = zeros((shape(codes)[0],1))
	
	for rowSample in range(0,shape(codes)[0]):
		dist = zeros((shape(ECOC)[0],1),float)
		for rowECOC in range(0,shape(ECOC)[0]):
			dist[rowECOC] = sqrt(float(sum((ECOC[rowECOC,:]-codes[rowSample,:])**2))) # euclidean distance
		
		y[rowSample] = dist.argmin() + 1
					
	return y
	
	
	
	
