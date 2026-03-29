# Construction de la structure
import numpy as np
class CNN:

    def __init__(self, filtersize, NbConvolution):
        self.NbConvolution = NbConvolution
        self.filtersize = filtersize
        self.filters = {}  # dictionnaire ayant en clé le numéro de la convolution et en valeur associé la matrice filtre
        self.resultconvul = {}
        for i in range(1, NbConvolution+1):
            self.filters[i] = np.ones((self.filtersize, self.filtersize))

    def extract_patches(self, img):
        H, W = img.shape # récupère les dimensions de la feature map

        kH, kW = (self.filtersize, self.filtersize)  # on récup la hauteur et largeur du flitre (tjrs carré)

        # calcule de la taille de la matrice de sortie :
        # le // c'est pour division entière

        out_H = H - kH + 1
        out_W = W - kW + 1

        # création d'un tableau avec que des zero dans lequel on mettera au fur et à mesure les matrices extraites

        patches = np.zeros((out_H, out_W, kH, kW))

        for i in range(out_H):
            for j in range(out_W):
                row = i
                col = j
                patches[i, j] = img[row:row + kH, col:col + kW]

        return patches

    def produit_sca(self,A,B):#A et B étant des matrices, pour convultion donner la matrice pour convolution t
        produit=0
        for i in range(0,A.shape[0]):
            for j in range(0,A.shape[0]):
                produit+= A[i,j]*B[i,j]
        return produit

    def ApplySca(self, MatriceDeMatrice, t):
        out_H, out_W, _, _ = MatriceDeMatrice.shape # donne le nombre de ligne et de colonne de la matrice de matrice

        mat_filtre = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                mat_filtre[i, j] = self.produit_sca(MatriceDeMatrice[i][j],self.filters[t])
        return mat_filtre

    def convolution(self, img, t):
        img = np.pad(img, pad_width=(self.filtersize - 1)//2, mode='constant', constant_values=0)  # padding
        featureINI = self.extract_patches(img)
        result = self.ApplySca(featureINI, t)
        return result

    def relu(self, img):
        return np.maximum(0, img)

    def pooling(self, img , pool_size=2, method = "Max"):
        H, W = img.shape
        pH, pW = pool_size, pool_size

        out_H = H // pH # j'ai cahnger le pooling car trop pussiant ca ne marchai pas avec des petites matrices
        out_W = W // pW

        output = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                row = i * 2
                col = j * 2
                fenetre = img[row:row + pH, col:col + pW]
                output[i, j] = np.max(fenetre)

        return output

    def flatten(self, feature_maps):
        return feature_maps.flatten() #focntion de numpy qui transforme en vecteur

        # Aplatit le tableau 3D en un vecteur 1D

        # Entrée  : feature_maps
        # Sortie  : vecteur

        pass


    def softmax(self, vecteur):
        exps = np.exp(vecteur - np.max(vecteur))  # fonction mathématique
        return exps / np.sum(exps)

        # Convertit les scores bruts en probabilités (somme = 1).
        pass


    def forward(self, img):

        # enchainement des convolutions et sauvegarde des résultats :

        self.resultconvul[0] = img
        for i in range(1, self.NbConvolution+1):
            img = self.convolution(img,i)
            img =self.relu(img)
            img = self.pooling(img)
            self.resultconvul[i]=img
        vecteur = self.flatten(img)
        proba = self.softmax(vecteur)
        return proba





Network = CNN(3,3)
img = np.random.randint(0, 10, (64,64))
print(Network.forward(img))









