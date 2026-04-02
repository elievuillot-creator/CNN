import CNN as CNN
import TraitementDonnées as td
import ReseauClassique2 as RC
import numpy as np

img = td.image_to_pixel_matrix("données/chien.png")


CNN = CNN.CNN(3, 3, [1,2,2])

  # 1 couche caché de 256 neuronnes, et 2 classes de sorties
print(CNN.UltimateForward(img))

def train(cnn, dataset, nb_epochs=10):
    """
    dataset : liste de tuples (chemin_image, label_one_hot)
              ex: [("données/chien.png", [1,0]), ("données/chat.png", [0,1]), ...]
    """
    for epoch in range(nb_epochs):
        total_loss = 0

        for chemin_img, label in dataset:
            # Chargement et forward pass CNN
            img = td.image_to_pixel_matrix(chemin_img)
            output = cnn.UltimateForward(img)

            # Calcul de la loss (cross-entropy)
            label_vec = np.array(label).reshape(1, -1)
            loss = -np.sum(label_vec * np.log(output + 1e-8))
            total_loss += loss

            # Backward pass via le MLP
            cnn.mlp.step(cnn.mlp._last_input, label_vec)

        print(f"Epoch {epoch+1}/{nb_epochs} | Loss moyenne : {total_loss/len(dataset):.4f}")





