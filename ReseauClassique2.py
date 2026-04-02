import numpy as np

class MLP:
    def __init__(self, neuronnes, app):
        """
        neuronnes : liste des tailles de couches, ex: [512, 256, 47]
                    neuronnes[0] = taille du vecteur d'entrée (sortie flatten du CNN)
                    neuronnes[-1] = nombre de classes en sortie
        app       : taux d'apprentissage
        """
        self.neuronnes = neuronnes
        self.app = app
        self.poids = []  # liste de matrices W, une par couche
        self.biais = []  # liste de vecteurs b, un par couche

        for i in range(len(neuronnes) - 1):
            n      = neuronnes[i]
            n_suiv = neuronnes[i + 1]
            # W de forme (n, n_suiv) : chaque colonne = poids vers un neurone suivant
            W = np.random.uniform(-np.sqrt(6 / n), np.sqrt(6 / n), size=(n, n_suiv))
            # b de forme (1, n_suiv) : un biais par neurone de la couche suivante
            B = np.zeros((1, n_suiv))
            self.poids.append(W)
            self.biais.append(B)

    def feedforward(self, x):
        """
        x : vecteur 1D ou (1, N) — typiquement la sortie aplatie (flatten) du CNN

        Couches cachées → sigmoid
        Couche de sortie → softmax (somme = 1, interprétable comme des probas)

        Retourne : liste de tous les vecteurs d'activation, entrée comprise
                   activations[0]  = vecteur d'entrée  (1, N)
                   activations[-1] = vecteur de sortie (1, nb_classes) somme = 1
        """
        # On force le vecteur en forme (1, N) pour les produits matriciels
        activation = x.reshape(1, -1)
        activations = [activation]  # on garde TOUTES les activations pour la backprop

        for i, (W, b) in enumerate(zip(self.poids, self.biais)):
            # z est un vecteur (1, n_suiv) : une valeur par neurone de la couche suivante
            z = np.dot(activation, W) + b
            # Dernière couche, softmax (somme = 1)
            if i == len(self.poids) - 1:
                activation = self._softmax(z)
            # Couches cachées, sigmoid
            else:
                activation = self._sigmoid(z)
            activations.append(activation)

        self._last_activations = activations
        return activations[-1]  # activations[-1] = vecteur de probas, somme = 1

    def delta_mat(self, label_vector, activations):
        """
        label_vector : vecteur cible (1, nb_classes), ex: [0,0,1,0,...] (one-hot)
        activations  : liste renvoyée par feedforward()

        La backprop calcule pour chaque couche un vecteur delta :
            delta = "à quel point chaque neurone a contribué à l'erreur"

        Retourne : liste de vecteurs delta, un par couche
                   deltas[-1] = erreur en sortie
                   deltas[1]  = gradient à renvoyer vers le CNN (couche d'entrée exclue)
        """
        # On initialise tous les deltas à zéro, même forme que les activations
        deltas = [np.zeros_like(a) for a in activations]

        # Couche de sortie
        # Erreur = différence entre ce qu'on voulait et ce qu'on a obtenu (vecteur)
        erreur_sortie = label_vector - activations[-1]
        # Delta sortie = erreur * dérivée sigmoid (neurone par neurone)
        deltas[-1] = erreur_sortie * self._sigmoid_deriv(activations[-1])

        # Couches cachées (on remonte de l'avant-dernière vers la deuxième)
        # On s'arrête à i=1 : deltas[0] (entrée) n'a pas de poids à mettre à jour
        for i in range(len(self.poids) - 1, 0, -1):
            # On propage le delta suivant à travers W^T pour obtenir l'erreur de cette couche
            # erreur_prop est un vecteur (1, n_couche_i)
            erreur_prop = np.dot(deltas[i + 1], self.poids[i].T)
            # Delta couche i = erreur propagée * dérivée sigmoid de cette couche
            deltas[i] = erreur_prop * self._sigmoid_deriv(activations[i])

        return deltas
        # Note : deltas[1] contient le gradient à rétropropager vers ton CNN

    def backwardpropagation(self, deltas, activations):
        """
        Met à jour les poids et biais de chaque couche.
        À appeler juste après delta_mat().

        Pour chaque couche i :
            ΔW = activation[i]^T · delta[i+1]   (produit externe de deux vecteurs → matrice)
            W  = W + lr * ΔW
            b  = b + lr * delta[i+1]
        """
        for i in range(len(self.poids)):
            # gradient est une matrice (n_i, n_i+1) : variation idéale de chaque poids
            # c'est le produit externe entre le vecteur activation et le vecteur delta
            gradient       = np.dot(activations[i].T, deltas[i + 1])
            self.poids[i] += self.app * gradient
            self.biais[i] += self.app * deltas[i + 1]


    def gradient_vers_cnn(self):
        """
        À appeler après delta_mat() pour récupérer le gradient
        à rétropropager dans les couches convolutives du CNN.

        Retourne un vecteur (1, taille_entree) que le CNN utilisera
        pour mettre à jour ses propres poids.
        """
        # deltas[1] · W[0]^T redonne un vecteur de la taille de l'entrée du MLP
        # ce vecteur EST le gradient qui remonte vers le flatten/pooling du CNN
        return np.dot(self._last_deltas[1], self.poids[0].T)

    def step(self, x, label_vector):
        self.feedforward(x)  # remplit self._last_activations
        self._last_deltas = self.delta_mat(label_vector, self._last_activations)
        self.backwardpropagation(self._last_deltas, self._last_activations)
        return np.dot(self._last_deltas[1], self.poids[0].T)


    def save(self, chemin="mlp_poids.txt"):
        blocs = []
        for W, b in zip(self.poids, self.biais):
            blocs.append(W.flatten())
            blocs.append(b.flatten())
        np.savetxt(chemin, np.concatenate(blocs))

    def load(self, chemin="mlp_poids.txt"):
        vecteur = np.loadtxt(chemin)
        idx = 0
        for i in range(len(self.neuronnes) - 1):
            n, n_suiv = self.neuronnes[i], self.neuronnes[i + 1]
            taille_w  = n * n_suiv
            self.poids[i] = vecteur[idx: idx + taille_w].reshape(n, n_suiv)
            idx += taille_w
            self.biais[i] = vecteur[idx: idx + n_suiv].reshape(1, n_suiv)
            idx += n_suiv


    def _sigmoid(self, z):
        # z est un vecteur, sigmoid est appliquée neurone par neurone
        return 1 / (1 + np.exp(-z))

    def _sigmoid_deriv(self, a):
        # a est DÉJÀ une activation sigmoid (pas un z brut)
        # la dérivée σ'(z) = σ(z)·(1-σ(z)) s'écrit simplement a·(1-a)
        return a * (1 - a)

    def _softmax(self, z):
        # la somme des sorties = 1
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps)