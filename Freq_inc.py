import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Charger une image à l'aide de PIL
image_path = 'DB/Train/images.jpg'
image = Image.open(image_path)

# Transformer l'image en un tenseur PyTorch
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convertir l'image PIL en un tenseur PyTorch
])

input_tensor = preprocess(image)
print(input_tensor.shape)  # Vérifier la forme du tenseur

nepochs=10
Nbsortie=5

crit = nn.MSELoss()

class Mod(nn.Module):
    def __init__(self,nhid):
        super(Mod, self).__init__()
        self.cnn = nn.Conv3d(in_channels=1, out_channels=Nbsortie, kernel_size=(3,3,3), stride=1, padding=(1,1,1))

    def forward(self,x):
        y=self.cnn(x)
        return y

def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    for epoch in range(nepochs):
        inputs, goldy = data
        optim.zero_grad()
        haty = mod(inputs)
        loss = crit(haty,goldy)
        nbatch += 1
        loss.backward()
        optim.step()
        print("err", loss)
    imgs = haty.cpu().detach().numpy()  # Conversion en tableau NumPy

    # Afficher chaque image des tenseurs de sortie
    for i in range(imgs.shape[2]):  # Parcourt la dimension de profondeur (depth)
        fig, axes = plt.subplots(1, imgs.shape[1], figsize=(12, 3))  # Création d'une figure avec des sous-graphiques
    
    # Parcourt les canaux de sortie pour afficher chaque image
        for j in range(imgs.shape[1]):
            axes[j].imshow(imgs[0, j, i, :, :], cmap='gray')  # Affichage de l'image avec cmap='gray' pour les images en niveaux de gris
            axes[j].axis('off')  # Désactive les axes
            axes[j].set_title(f'Channel {j+1}')  # Définit un titre pour chaque image
    
        plt.show() 

mod=Mod(h)
print("nparms",sum(p.numel() for p in mod.parameters() if p.requires_grad),file=sys.stderr)
train(mod)