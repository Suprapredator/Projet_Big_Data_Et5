import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import math as mt;
import copy

### REGARDER NUMPY PLS "Ca fait le café" -Quentin 2018

def affinerContours(data, index):
	image = copy.deepcopy(data)

	# Détermination es valeurs moyennes
	moyennes = [0, 0, 0];
	for i in range(32):
		for j in range (32):
			for k in range (3):
				moyennes[k] += data['X'][i, j, k, index]
		
	
	for k in range (0, 3):
		moyennes[k] /= 32*32
	
	# Création de la nouvelle image
	for i in range(32):
		for j in range (32):
			couleur = [0, 0, 0]
			for k in range (3):
				if data['X'][i, j, k, index] > moyennes[k]:
					couleur[k] = 255
				else:
					couleur[k] = 0

			if sum(couleur) > 255 :
				image['X'][i, j, :, index] = [0, 0, 0]
			else:
				image['X'][i, j, :, index] = [255, 255, 255]

	return image

if __name__ == "__main__":

    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')

    image_idx = 62;
    nouvelImage = affinerContours(train_data, image_idx)
    print('Label:', train_data['y'][image_idx])
    plt.imshow(train_data['X'][:, :, :, image_idx])
    plt.show()
    plt.imshow(nouvelImage['X'][:, :, :, image_idx])
    plt.show()
