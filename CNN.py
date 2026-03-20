# Construction de la structure

class CNN:

    def __init__(self):


    def convolution(self, image, filters, filter_size):

        # image : en tableau de pixel
        # filters : nombre de filtres
        # filter_size : taillle des filtres (donc genre 3 pour 3x3)


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
x = Network.NeuralNetworkClassique(x, out_features=256)
x = Network.softmax(x)

