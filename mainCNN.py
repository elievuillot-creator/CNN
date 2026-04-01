import CNN as CNN
import TraitementDonnées as td


img = td.image_to_pixel_matrix("données/chien.png")


CNN = CNN.CNN(3, 3, [1,2,2])
print(CNN.UltimateForward(img))

