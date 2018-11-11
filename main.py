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
import apprentissage as app;
import pretraitement as pre;

def printPicture(data, index, RVB):
    print(data['X'][:,:,RVB,index])

if __name__ == "__main__":
    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')
    
    moyennes = moy.getMoyennes(train_data)
    
    print('Résultat par moyenne:')
    moy.test(moyennes, train_data, 100)
    print('Résultat par Sklearn:')
    app.test(train_data, 100)
    