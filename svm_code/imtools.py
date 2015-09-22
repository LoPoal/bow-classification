import os

# set of useful tools for image processing

def getImlist(path,ext):
	# returns a list of filenames (absolute, relative) for all 'ext' images in a directory
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.'+ext+'')],  [f for f in os.listdir(path) if f.endswith('.'+ext+'')]
