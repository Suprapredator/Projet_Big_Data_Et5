# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import math as mt;
import moyenne as moy;
import pretraitement as pre;

def printPicture(data, index, RVB):
    print(data['X'][:,:,RVB,index])

if __name__ == "__main__":
    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')
    
    #for i in range(100):
    #    train_data['X'][:,:,:,i] = pre.affinerContours(train_data, i)
    
    moyennes = moy.getMoyennes(train_data)
    
    #plt.imshow(moyennes[1])
    #plt.show()
    
    moy.test(moyennes, train_data, 1000)
    
    #img = (moy.calculerImageMoyenneClasseX(2,train_data))
    #print(img)
    #plt.imshow(img)
    #plt.show()