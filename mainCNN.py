import CNN as module_CNN
import TraitementDonnées as td
import ReseauClassique2 as RC
import numpy as np
import os
from PIL import Image


def charger_dataset(dossier_chien, dossier_chat, limite=500, taille=(64, 64)):
    dataset = []

    images_chien = [f for f in os.listdir(dossier_chien)
                    if f.endswith(".jpg") or f.endswith(".png")][:limite]
    for nom_fichier in images_chien:
        chemin = os.path.join(dossier_chien, nom_fichier)
        try:
            img = Image.open(chemin).convert("RGB").resize(taille)
            img = np.array(img)
            dataset.append((img, [1, 0]))
        except:
            continue

    images_chat = [f for f in os.listdir(dossier_chat)
                   if f.endswith(".jpg") or f.endswith(".png")][:limite]
    for nom_fichier in images_chat:
        chemin = os.path.join(dossier_chat, nom_fichier)
        try:
            img = Image.open(chemin).convert("RGB").resize(taille)
            img = np.array(img)
            dataset.append((img, [0, 1]))
        except:
            continue

    np.random.shuffle(dataset)
    print(f"Dataset chargé : {len(dataset)} images")
    return dataset


def train(cnn, dataset, nb_epochs=10):
    for epoch in range(nb_epochs):
        total_loss = 0
        for img, label in dataset:
            output = cnn.UltimateForward(img)
            label_vec = np.array(label).reshape(1, -1)
            loss = -np.sum(label_vec * np.log(output + 1e-8))
            total_loss += loss
            cnn.mlp.step(cnn._last_flat, label_vec)



# --- Main ---
DOSSIER_CHIEN = r'C:\Users\Vuillot\Downloads\PetImages\Dog'
DOSSIER_CHAT  = r'C:\Users\Vuillot\Downloads\PetImages\Cat'

dataset = charger_dataset(DOSSIER_CHIEN, DOSSIER_CHAT, limite=500)

cnn = module_CNN.CNN(3, 3, [1, 2, 2])
train(cnn, dataset, nb_epochs=20)