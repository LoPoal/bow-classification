__author__ = 'miquel'

import cv2
import leargist
from PIL import Image
import warnings
def gist(img):
    warnings.filterwarnings("ignore")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pilImage= Image.fromarray(img)
    #Note that we put 1.5 to avoid having 960 dimensions per feature. Using 1 it just returns 60.
    des = leargist.color_gist(pilImage,1)
    return des


