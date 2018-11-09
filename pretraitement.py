import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import math as mt;
import copy

def affinerContours(data, index):
	#tab = [0, 0, 0]
	image = copy.deepcopy(data)

	background = 0;

	# Détermination 
	for i in range(0, 32):
		for j in range (0, 32):
			for k in range (0, 3):
				background += data['X'][i, j, k, index]
		

	background /= 32*32*3
	print("bg :", background)
	for i in range(0, 32):
		for j in range (0, 32):
			couleur = 0
			for k in range (0, 3):
				couleur += data['X'][i, j, k, index]

			if abs(background-couleur) < 70 : couleur = 255
			else: couleur = 0
			image['X'][i, j, :, index] = [couleur, couleur, couleur]

	return image


if __name__ == "__main__":

    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')

    image_idx = 2;
    nouvelImage = affinerContours(train_data, image_idx)
    print('Label:', train_data['y'][image_idx])
    plt.imshow(train_data['X'][:, :, :, image_idx])
    plt.show()
    plt.imshow(nouvelImage['X'][:, :, :, image_idx])
    plt.show()
