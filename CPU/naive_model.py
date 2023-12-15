import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imgs_to_vid import slice_video
from torchsummary import summary
import cv2

'''
# Charger une image à l'aide de slice_vid
frames, frame_width, frame_height, fps, fourcc = slice_video('video_test.mp4')
# Isole uniquement 5 frames pour la partie débuggage du réseau
frames=frames[:11]
'''

def img2tens(frames, mode='train', test_size=0.3):
    if mode=='train':
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

        X_frames=[]
        for k in range(len(X)-1):
            pair=[X[k],X[k+1]]
            X_frames.append(pair)

        X_frames=np.array(X_frames)
        y_frames=np.array(y)
        X_train_frames, X_test_frames, y_train_frames, y_test_frames = train_test_split(X_frames, y_frames, test_size=test_size)
        X_train_tensor=torch.tensor(X_train_frames).float()
        y_train_tensor=torch.tensor(y_train_frames).float()
        X_test_tensor=torch.tensor(X_test_frames).float()
        y_test_tensor=torch.tensor(y_test_frames).float()
        return X_train_tensor,y_train_tensor, X_test_tensor, y_test_tensor
    
    if mode=='forward':
        X_frames=[]
        for k in range(len(frames)-1):
            pair=[frames[k],frames[k+1]]
            X_frames.append(pair)

        X_frames=np.array(X_frames)
        X_tensor=torch.tensor(X_frames).float()
        return X_tensor
    
    print("Veuillez entrer un mode correcte pour l'utilisation du modèle (train ou forward)")
    return 0
        

class Mod(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mod, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        #Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x1 = self.encoder(x)
        x1 = 0.5*(x1[:,:,0,:,:]+x1[:,:,1,:,:])
        x1 = x1.unsqueeze(2)
        x2 = self.bottleneck(x1)
        x3 = self.decoder(torch.cat([x1, x2], dim=1)) #Il faut faire l'interpolation dans l'espace latent avant de l'envoyer dans le décodeur
        x4 = x3.squeeze(2)
        return x4

def test(model, testx, testy):
    crit = nn.MSELoss()
    inputs, goldys = testx , testy
    goldys = goldys.permute(0,3,1,2)
    haty = model(inputs)
    val_loss = crit(haty,goldys)
    return val_loss.item()
    
def train(model, data, target, testx, testy, nepochs):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    inputs, goldys = data , target
    goldys = goldys.permute(0,3,1,2)
    ITER=[]
    LOSS=[]
    VAL_LOSS=[]
    i=0
    for epoch in range(nepochs):
        optim.zero_grad()
        haty = model(inputs)
        loss = crit(haty,goldys)
        print(f"err de : {loss} à la {epoch+1}e epoch")
        i+=1
        ITER.append(i)
        LOSS.append(loss.item())
        VAL_LOSS.append(test(model,testx,testy))
        loss.backward()
        optim.step()
        
    #Affichage de la courbe de loss train et val 
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Perte (LOSS)', color=color)
    ax1.plot(ITER, LOSS, label='Perte (LOSS)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss de validation', color=color)
    ax2.plot(ITER, VAL_LOSS, label='Loss de validation (VAL_LOSS)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Courbes de loss au cour des epochs')
    fig.tight_layout()
    plt.savefig('loss_curves.jpg')
    plt.show()

'''
mod=Mod(3,3)
summary(mod,(3, frame_height, frame_width, 3))    
  
nepochs=5
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = img2tens(frames)

train(mod,X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, nepochs)

output = mod(X_train_tensor)
'''

def tens2img(output_tens) :
    output_tens = output_tens.permute(0,2,3,1)
    output_np = output_tens.cpu().detach().numpy()

    # Échelle des valeurs pour les images en uint8 (0-255)
    scaled_output = ((output_np - output_np.min()) / (output_np.max() - output_np.min()) * 255).astype(np.uint8)
    return(scaled_output)

'''
scaled_output = tens2img(output)

# Afficher chaque image des tenseurs de sortie

plt.figure(figsize(64,4))
plt.subplot(2,2,1)
plt.imshow(scaled_output[0, :, :, :], cmap='gray')
plt.title('Image 1')

plt.subplot(2,2,2)
plt.imshow(scaled_output[1, :, :, :], cmap='gray')
plt.title('Image 2')

plt.subplot(2,2,3)
plt.imshow(scaled_output[2, :, :, :], cmap='gray')
plt.title('Image 3')

plt.subplot(2,2,4)
plt.imshow(scaled_output[3, :, :, :], cmap='gray')
plt.title('Image 4')

plt.tight_layout()
plt.imsave('exemples_results_apprentissage.jpg')

plt.show()
'''