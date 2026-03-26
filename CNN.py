# Construction de la structure
import numpy as np
class CNN:

    def __init__(self, filtersize, NbConvolution):
        self.NbConvolution = NbConvolution
        self.filtersize = filtersize
        self.filters = {}  # dictionnaire ayant en clé le numéro de la convolution et en valeur associé la matrice filtre
        for i in range(1, NbConvolution+1):
            self.filters[i] = np.ones((self.filtersize, self.filtersize))

    def extract_patches(self, img, stride=1):
        H, W = img.shape # récupère les dimensions de la feature map

        kH, kW = (self.filtersize, self.filtersize)  # on récup la hauteur et largeur du flitre (tjrs carré)

        # calcule de la taille de la matrice de sortie :
        # le // c'est pour division entière

        out_H = (H - kH) // 2
        out_W = (W - kW) // 2

        # création d'un tableau avec que des zero dans lequel on mettera au fur et à mesure les matrices extraites

        patches = np.zeros((out_H, out_W, kH, kW))

        for i in range(out_H):
            for j in range(out_W):
                row = i * stride
                col = j * stride
                patches[i, j] = feature_map[row:row + kH, col:col + kW]

        return patches


    def convolution(self, img, filtre):
        img = np.pad(img, pad_width=(self.filtersize - 1)/2, mode='constant', constant_values=0)
        featureINI = self.extract_patches(img)





        # sortie : tableau "feature_maps"

        pass

    def relu(self, feature_maps):
        # transforme les valeurs négatives en 0
        pass

    def pooling(self, feature_maps, size=2, methode = "MAX"):

        # prend en entrée la feature_maps après passage de la RELU
        # garde le max des 4 pixel pris (car 2x2)

        pass

    def flatten(self, feature_maps):

        # Aplatit le tableau 3D en un vecteur 1D

        # Entrée  : feature_maps
        # Sortie  : vecteur

        pass

    def NeuralNetworkClassique(self, vecteur):
        pass

    def softmax(self, vecteur):

        # Convertit les scores bruts en probabilités (somme = 1).

        pass




#exemple main :

Network = CNN()

x = Network.convultion(Image, filters = 32, filter_size = 3)
x = Network.pooling(x)
x = Network.relu(x)
x = Network.flatten(x)
x = Network.NeuralNetworkClassique(x)
x = Network.softmax(x)

