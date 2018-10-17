# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import math as mt;

def printPicture(data, index, RVB):
    print(data['X'][:,:,RVB,index])
            
def calculerImageMoyenneClasseX(X,data):
    nbre = 0
    
    for i in range(len(data['y'])):
        if data['y'][i] == X:
            nbre += 1
    
    ok = True
    for i in range(len(data['y'])):
        if data['y'][i] == X:
            if ok:
                average = data['X'][:,:,:,i]/nbre
                ok = False
            else:
                average = average + data['X'][:,:,:,i]/nbre
            #print(average[0][0])
    
    #print(nbre)
    return average/255

def floatMatriceToIntegerMatrice(M):
    for i in range(32):
       for j in range(32):
          for k in range(3):
              M[i][j][k] = mt.modf(M[i][j][k])[1]
              
    return M

if __name__ == "__main__":

    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')

    image_idx = 0
    #plt.imshow(train_data['X'][:, :, :, image_idx])
    #printPicture(train_data, 0, 0)
    #printPicture(train_data, 1, 0)
    #print(train_data['X'][:, :, :, 0])
    #print(train_data['X'][:, :, :, 0])

    test = train_data['X'][:, :, :, 0] + train_data['X'][:, :, :, 0]
    #print(test)
    img = (calculerImageMoyenneClasseX(1,train_data))
    print(img)
    plt.imshow(img)
    plt.show()