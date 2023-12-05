import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from slice_vid import slice_video
import cv2
import torch.nn.functional as F

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

X_frames=np.array(X)
y_frames=np.array(y)
X_frames_tensor=torch.tensor(X_frames).float()
y_frames_tensor=torch.tensor(y_frames).float()

print(X_frames_tensor.shape,y_frames_tensor.shape)

nepochs=10
Nbsortie=2

crit = nn.MSELoss()

class Mod(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mod, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        #Décodeu
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.bottleneck(x1)
        x3 = self.decoder(torch.cat([x1, x2], dim=1))
        x4 = 0.5*(x3[:,:,0,:,:]+x3[:,:,1,:,:]) 
        return x4

def train(mod,data,target,nepochs):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    inputs, goldys = data , target
    for epoch in range(nepochs):
        optim.zero_grad()
        loss=0
        for k in range(len(data)-1):
            input=inputs[k:k+2].unsqueeze(0)
            haty = mod(input.permute(0, 4, 1, 2, 3))
            goldy = goldys[k].unsqueeze(0)
            goldy = goldy.permute(0,3,1,2)
            loss += crit(haty,goldy)
        loss/=len(data)
        print("err", loss)
        loss.backward()
        optim.step()

mod=Mod(3,3)
#print("nparms",sum(p.numel() for p in mod.parameters() if p.requires_grad),file=sys.stderr)
train(mod,X_frames_tensor, y_frames_tensor, 10)
output = mod(X_frames_tensor[0:2].unsqueeze(0).permute(0, 4, 1, 2, 3))
output=output.permute(0,2,3,1)
output_np = output.cpu().detach().numpy()

# Échelle des valeurs pour les images en uint8 (0-255)
scaled_output = ((output_np - output_np.min()) / (output_np.max() - output_np.min()) * 255).astype(np.uint8)
print(scaled_output)
# Afficher chaque image des tenseurs de sortie
plt.imshow(scaled_output[0, :, :, :], cmap='gray')
plt.title('Image 1')
plt.show()