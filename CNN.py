# Construction de la structure
import numpy as np
class CNN:

    def __init__(self, filtersize, NbConvolution, NbFilters):
        self.NbConvolution = NbConvolution
        self.filtersize = filtersize
        self.filters = {}
        self.resultconvul = {}
        self.biases = {}  #  un biais scalaire par couche de convolution
        self.NbFilters = NbFilters
        self.Deep = []
        self.Deep.append(3)
        self.Biais = {}

        # calcul des nb de cannaux par filtres de chaque convolution :
        for i in range(1, NbConvolution):
            self.Deep.append(self.NbFilters[i-1])

        # Initialisation des filtres :

        for i in range(0, NbConvolution):
            self.Biais[i] = []
            self.filters[i] = []
            for j in range(self.NbFilters[i]):
                self.filters[i].append([])
                self.Biais[i].append(1)
            for k in range(self.NbFilters[i]):
                for p in range(self.Deep[i]):
                    self.filters[i][k].append(np.random.rand(self.filtersize, self.filtersize))


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

    def ApplySca(self, MatriceDeMatrice, FILTRE):
        out_H, out_W, _, _ = MatriceDeMatrice.shape # donne le nombre de ligne et de colonne de la matrice de matrice

        mat_filtre = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                mat_filtre[i, j] = self.produit_sca(MatriceDeMatrice[i][j],FILTRE)
        return mat_filtre

    def convolution(self, img, FILTRE):
        img = np.pad(img, pad_width=(self.filtersize - 1)//2, mode='constant', constant_values=0)
        featureINI = self.extract_patches(img)
        result = self.ApplySca(featureINI, FILTRE)
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


    def Forward2(self, img, filter):
        img = self.convolution(img, filter)
        img = self.relu(img)
        img = self.pooling(img)
        return img

    def passage_dim_3D_2D(self, liste_de_matrice):
        H, W = liste_de_matrice[0].shape  # récuper la taille des matric filtré
        nouv_matrice_2D = np.zeros((H, W))
        for j in range(len(liste_de_matrice)):
            nouv_matrice_2D += liste_de_matrice[j]
        return nouv_matrice_2D # renvoie une matrice 2D qui est est la somme des matrcie filtré

    def ConvoAgre(self, Mat, NumConvo):
        liste = []
        list1 = []
        Result = []
        for i in range(self.Deep[NumConvo]):
           liste.append(Mat[:, :, i])
        for k in range(0, self.NbFilters[NumConvo]):  # pour un filtre
            for j in range(self.Deep[NumConvo]):  # pour une couche
                list1.append(self.Forward2(liste[j], self.filters[NumConvo][k][j]))
            Result.append(self.pooling(self.relu(self.passage_dim_3D_2D(list1) + self.Biais[NumConvo][k])))  # donc met dans les résultat de la convolution la matrice 2D renvoyée par le premier filtre
        Result = np.stack(Result, axis=2)
        return Result
            # donc list1 contient les trois matrices résultants du filtre que la fct de albin doit mettre en 1

    def flatten(self, feature_maps):
        return feature_maps.flatten()


    def UltimateForward(self, img):
        n = self.NbConvolution
        for i in range(0,n):
            self.resultconvul[i] = self.ConvoAgre(img, i)
            img = self.resultconvul[i]
        vector_flat = self.flatten(img)
        return img





    def softmax(self, vecteur):
        exps = np.exp(vecteur - np.max(vecteur))  # fonction mathématique
        return exps / np.sum(exps)

        # Convertit les scores bruts en probabilités (somme = 1).
        pass




