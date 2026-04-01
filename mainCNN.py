import CNN as CNN
import TraitementDonnées as td
import ReseauClassique2 as RC


img = td.image_to_pixel_matrix("données/chien.png")


CNN = CNN.CNN(3, 3, [1,2,2])

  # 1 couche caché de 256 neuronnes, et 2 classes de sorties
print(CNN.UltimateForward(img))

