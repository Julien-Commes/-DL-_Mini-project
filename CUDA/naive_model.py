import numpy as np
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imgs_to_vid import slice_video
#from torchsummary import summary

'''
# Charger une image à l'aide de slice_vid
frames, frame_width, frame_height, fps, fourcc = slice_video('video_test.mp4')
# Isole uniquement 5 frames pour la partie débuggage du réseau
#frames=frames[:70]
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        X_train_tensor,y_train_tensor, X_test_tensor, y_test_tensor = X_train_tensor.to(device), y_train_tensor.to(device), X_test_tensor.to(device), y_test_tensor.to(device)
        return X_train_tensor,y_train_tensor, X_test_tensor, y_test_tensor
    
    if mode=='forward':
        X_frames=[]
        for k in range(len(frames)-1):
            pair=[frames[k],frames[k+1]]
            X_frames.append(pair)

        X_frames=np.array(X_frames)
        X_tensor=torch.tensor(X_frames).float()
        X_tensor = X_tensor.to(device)
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

def test(model, testloader):
    model.train(False)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    val_ssim = 0.
    val_psnr = 0.
    val_loss, nbatch = 0., 0
    crit = nn.MSELoss()
    for data in testloader :
        inputs, goldys = data
        goldys = goldys.permute(0,3,1,2)
        haty = model(inputs)
        loss = crit(haty,goldys)
        val_ssim += ssim(haty, goldys).item()
        val_psnr += psnr(haty,goldys).item()
        val_loss += loss.item()
        nbatch += 1
    val_ssim /= float(nbatch)
    val_psnr /= float(nbatch)
    val_loss /= float(nbatch)
    model.train(True)
    return val_loss, val_ssim, val_psnr
    
def train(model, trainloader, testloader, nepochs):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    ITER=[]
    LOSS=[]
    VAL_LOSS=[]
    VAL_SSIM=[]
    VAL_PSNR=[]
    for epoch in range(nepochs):
        val_loss, val_ssim, val_psnr = test(model, testloader)
        totloss, nbatch = 0., 0
        for data in trainloader :
            inputs, goldys = data
            goldys = goldys.permute(0,3,1,2)
            optim.zero_grad()
            haty = model(inputs)
            loss = crit(haty,goldys)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        ITER.append(epoch)
        VAL_LOSS.append(val_loss)
        VAL_SSIM.append(val_ssim)
        VAL_PSNR.append(val_psnr)
        LOSS.append(totloss)
        print(f"err de : {totloss} à la {epoch+1}e epoch")
        
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
    
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.set_ylabel('SSIM de validation', color=color)
    ax3.plot(ITER, VAL_SSIM, label='SSIM de validation (VAL_SSIM)', color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    
    ax4 = ax1.twinx()
    color = 'tab:orange'
    ax4.set_ylabel('PSNR de validation', color=color)
    ax4.plot(ITER, VAL_PSNR, label='PSNR de validation (VAL_PSNR)', color=color)
    ax4.tick_params(axis='y', labelcolor=color)

    plt.title('Courbes de loss au cour des epochs')
    fig.tight_layout()
    plt.savefig('loss_curves.jpg')
    plt.show()

'''
mod=Mod(3,3)
mod=mod.to(device)
#summary(mod,(3, frame_height, frame_width, 3))    
nepochs=50

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = img2tens(frames)

trainds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=7, shuffle=False)
testds = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
testloader = torch.utils.data.DataLoader(testds, batch_size=7, shuffle=False)

train(mod, trainloader, testloader, nepochs)

Frames_tensor = img2tens(frames, mode='forward')
forwardds = torch.utils.data.TensorDataset(Frames_tensor)
forwardloader = torch.utils.data.DataLoader(forwardds, batch_size=20, shuffle=False)

for data in forwardloader:
    inputs = data[0]
    output = mod(inputs)
'''

def tens2img(output_tens) :
    output_tens = output_tens.permute(0,2,3,1)
    output_np = output_tens.cpu().detach().numpy()

    # Échelle des valeurs pour les images en uint8 (0-255)
    scaled_output = ((output_np - output_np.min()) / (output_np.max() - output_np.min()) * 255).astype(np.uint8)
    return(scaled_output)

'''
scaled_output = tens2img(output)

# Afficher des images de sortie
plt.figure(figsize=(64,4))
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
plt.show()
'''