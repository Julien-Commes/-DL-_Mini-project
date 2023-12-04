import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from slice_vid import slice_video

# Charger une image à l'aide de slice_vid
frames=slice_video('video_test.mp4')

# Isole uniquement 5 frames pour la partie débuggage du réseau
frames=frames[:5]

if len(frames)<3:
    print("La vidéo ne contient pas assez de frames")
    exit()

if len(frames)%2==0:
    frames.pop()

y=[]
X=[]
i=0
for frame in frames :
    i+=1
    if i%2!=0:
        X.append(frame)
    else :
        y.append(frame)

X_frames_tensor=torch.tensor(X)
y_frames_tensor=torch.tensor(y)

print(X_frames_tensor.shape,y_frames_tensor.shape)

nepochs=10
Nbsortie=2

crit = nn.MSELoss()

class Mod(nn.Module):
    def __init__(self):
        super(Mod, self).__init__()
        self.cnn = nn.Conv3d(in_channels=1, out_channels=Nbsortie, kernel_size=(3,3,3), stride=1, padding=(1,1,1))

    def forward(self,x):
        y=self.cnn(x)
        return y

def train(mod,data,target):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    for epoch in range(nepochs):
        inputs, goldy = data , target
        optim.zero_grad()
        haty = mod(inputs)
        loss = crit(haty,goldy)
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

mod=Mod()
#print("nparms",sum(p.numel() for p in mod.parameters() if p.requires_grad),file=sys.stderr)
#train(mod)
print(mod(X_frames_tensor))