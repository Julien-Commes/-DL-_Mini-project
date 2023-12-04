from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Charger une image à l'aide de PIL
image_path = 'DB/Train/20231108_232546.jpg'
image = Image.open(image_path)

# Transformer l'image en un tenseur PyTorch
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convertir l'image PIL en un tenseur PyTorch
])

input_tensor = preprocess(image)
print(input_tensor.shape)  # Vérifier la forme du tenseur

Cnn_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(3,3), stride=1, padding=(1,1))
output_tensor = Cnn_layer(input_tensor)

print(output_tensor.shape)
imgs = output_tensor.cpu().detach().numpy()  # Conversion en tableau NumPy

    # Afficher chaque image des tenseurs de sortie
 # Création d'une figure avec des sous-graphiques
    
    # Parcourt les canaux de sortie pour afficher chaque image
plt.imshow(imgs[0, :, :], cmap='gray')  # Affichage de l'image avec cmap='gray' pour les images en niveaux de gris
plt.show() 