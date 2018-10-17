# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;

if __name__ == "__main__":

    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')

    image_idx = 0
    print('Label:', train_data['y'][image_idx])
    plt.imshow(train_data['X'][:, :, :, image_idx])
    plt.show()