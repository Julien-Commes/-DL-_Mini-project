# Machine Learning & Differentiable Programming Mini-project
## VFI (Video Frame Interpolation)

### Bibliothèques et utilisation

#### Bibliothèques

* numpy
* pytorch
* scikitlearn
* openCV
* os
* argparse
* torchmetrics
* torchsummary (optionnel)

#### Utilisation 

Pour augmenter la fréquence d'une vidéo, exécuter le fichier vfi_end2end.py avec le chemin vers la vidéo (disponible dans /CPU ou /CUDA) :

```bash
python3 vfi_end2end.py ../DB/video_test.mp4
```

(Il peut être nécessaire de modifier la taille des batchs dans le fichier selon les capacités de la machine)

Pour faire uniquement un entrainement et s'amuser avec les paramètres du modèle, exécuter le fichier naive_model.py après avoir "unquote" le code d'éxécution sur celui-ci.

### Contexte

L'objectif de ce projet est de créer un modèle basé sur un ou une combinaison de réseaux de neurones (MLP, LSTM, ConvNet, Transformer...), de l'entrainer sur une base de données, évaluer et analyser ses performances puis itérer pour l'améliorer dans sa tâche ou le remplacer par un autre modèle.

La tâche qui a été choisie ici est celle de l'interpolation d'image en vidéo ou Video Frame Interpolation (VFI). Celle-ci consiste à déterminer une nouvelle image entre 2 images adjacentes d'une vidéo. Parmi les nombreuses applications on peut notamment citer l'augmentation du nombre de frame (dans la restauration de vieilles vidéos ou pour fluidifier des vidéos en stop motion par exemple), les ralentis de vidéo ou encore la fluidification du flux vidéo (télévision, streaming).

### Travaux actuels

Une méthode de VFI couramment utilisée aujourd'hui est la recherche du 'flux optique' des objets présents sur l'image. L'idée générale est que l'on cherche à déterminer le mouvement d'un objet (groupe de pixels) entre plusieurs images successives afin de le déplacer à une localisation intermédiaire. Les algorithmes généralement utilisés pour identifier et applique le flux optique dans des systèmes comme des écrans de télévision sont déterministes, cependant depuis 2017 des algorithmes de réseaux neuronaux profonds ont été développés afin de déterminer le flux optique. 

Une autre approche appelée "flow-free" consiste à appliquer des convolutions et fusionner les images adjacentes. Nous allons ici nous inspirer de ces approches pour notre modèle.

### Modèle 

Dans la phase de recherche d'un modèle à tester, deux pistes ont été explorées. La première s'appuie sur la publication [RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/pdf/2011.06294v11.pdf) par Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou. Celui-ci propose une utilisation du réseau IFNet afin de trouver le flux optique avec un apprentissage non supervisé. Il établi l'état de l'art sur plusieurs bases de données et est rapide en temps d'éxecution. De plus le code est disponible [en ligne](https://github.com/megvii-research/ECCV2022-RIFE). 
Néanmoins cette piste a dû être écartée dans un premier temps pour plusieurs raisons. En effet, le modèle semblait trop complexe pour une première itération. De plus, une forte contrainte (que l'on retrouvera de manière récurrente dans ce projet) était la taille du modèle et la possibilité de l'éxécuter sur une machine ayant des performances modestes.

Une seconde piste a été trouvée grâce à la publication [The U-Net based GLOW for Optical-Flow-free Video Interframe Generation](https://arxiv.org/pdf/2103.09576.pdf) par Saem Park, Donghoon Han et Nojun Kwak. La méthode proposée consiste à utiliser un réseau neuronal profond générant les images intermédiaires par des convolutions basé sur l'architecture d'U-Net. 
Bien que la publication propose des méthodes non-triviales pour la génération des images et l'apprentissage. L'utilisation du réseau U-net semble être une bonne base de départ pour notre tâche. 

En effet, l'architecture d'U-Net semble à priori présenter plusieurs avantages dans la tâche de VFI. L'architecture U-Net présente une connexion directe entre les encodeurs et les décodeurs, ce qui permet une transmission directe des informations spatiales à différentes échelles et pourrait permettre de conserver la cohérence spatiale entre les images d'entrée et les images interpolées. De plus, la structure en U de l'architecture U-Net pourrait permettre une intégration efficace des informations temporelles. Les blocs d'encodeurs peuvent extraire des caractéristiques de haute résolution à partir des images d'entrée, tandis que les blocs de décodeurs peuvent les utiliser pour générer des images interpolées. Ses couches profondes pourraient permettent d'apprendre des représentations plus complexes alors que la forme générale des objets serait captée par les couches de surface. Enfin, notre modèle ne doit pas perdre d'information entre l'entrée et la sortie afin d'éviter la création d'effets de flou ce qui est le cas de l'architecture U-Net.

D'un point de vue pratique, l'architecture U-Net est plutôt simple à comprendre et à implémenter ce qui est avantageux pour une première itération. De plus, il est facilement possible d'imaginer des variations de l'architecture ce qui est intéressant au vue des problèmatiques du projet (en particulier la taille du modèle). 

L'architecture d'origine de U-Net est la suivante (image récupérée de la publication [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) par Olaf Ronneberger, Philipp Fischer et Thomas Brox) ![U-Net architecture](/images/u-net-architecture-1024x682.png "U-Net architecture")

### Implémentation

Une implémentation simple de l'architecture U-Net pour pytorch est disponible sur la page [UNET Implementation in PyTorch — Idiot Developer](https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201). Cette implémentation a inspirée celle utilisée pour ce projet. 

Néanmoins, les premiers essais d'utilisation du réseau pour un apprentissage sur une vidéo courte (moins de 6 secondes) se sont tous soldés par un Kill du process par la machine. Le modèle contenait beaucoup trop de paramètres et devait tourner sur une machine sans GPU. Il a fallu donc imaginer une version encore simplifiée de l'architecture. Les couches profondes ont donc été successivement retirées afin de réduire le nombre de paramètres jusqu'à arriver à un seul bloc de convolutions dans l'encodeur et dans le décodeur au lieu de 4. 
De plus, l'architecture U-Net n'est pas utilisée à l'origine pour des tâches d'interpolation vidéo mais pour de la segmentation d'image. Il a donc fallu adapter les couches de convolutions pour une donnée vidéo et penser à quand fusionner les images adjacentes dans notre architecture. Dans le modèle retenu, celle-ci à lieu après le passage des images adjacentes dans le bloc encodeur. 

Exemple d'images avec fusion après le passage dans le décodeur (50 epochs):
![outfuse](/images/figures/Figure_1_50ep_MSE_Adam_1.4_BS7_outfuse.png "outfuse")

Exemple d'images avec fusion après le passage dans l'encodeur (50 epochs):
![infuse](/images/figures/Figure_1_50ep_MSE_Adam_1.1_BS7.png "infuse")

(Les résultats étaint prévisibles car on imagine bien que la fusion va apporter du flou et des imperfections qui ne seront pas corrigé en sortie du décodeur)

Une question restante porte sur la méthode d'apprentissage et le domaine d'application du modèle : Est ce que l'on veut que faire apprendre à notre modèle l'interpolation sur une vidéo puis lui demander d'en augmenter le nombre d'images ou est ce qu'il faut entrainer le modèle sur un grand corpus de vidéos afin qu'il puisse généraliser à toutes les vidéos. Pour des raisons de temps d'apprentissage et parce que le modèle semble trop peu volumineux, on a choisi ici de commencer par un modèle qui apprends sur une vidéo puis augmente son nombre d'images. 

### Apprentissage 

#### Création des données d'apprentissages

On cherche à apprendre générer des images intermédiaires entre 2 images adjacentes d'une vidéo. Il faut donc pour l'apprentissage du modèle prendre 3 images successives, en extraire l'image centrale comme étant la vérité terrain et prendre la paire d'image restant comme notre tenseur d'entrée. Une vidéo comprenant N frames donnera alors (N-1)//2 paires car une même image d'entrée appartient à 2 paires (sauf celles de début et de fin de la vidéo).

#### Optimisation et Loss

Deux optimiseurs ont étés testés : Adam et la SGD. L'utilisation de la SGD a tendance à obscurcir l'image voir obtenir une image noire et reste bloqué à ce minimum local. Adam a donc été privilégié. 

Quelques générations d'images en utilisant la SGD (50 epochs): ![SGD-optim](/images/figures/Figure_1_50ep_PSNR_SGD_1.1_BS7.png "SGD optim")

La problèmatique de la métrique d'évaluation du modèle est aussi importante. En effet, celle-ci doit normalement être utilisée pour estimer la capacité du modèle à généraliser sur des données sur lesquels il n'apprend pas. Ici le problème est que bien qu'il n'apprends sur les données d'évaluation, sur une vidéo courte (moins de 10 secondes) et surtout avec une fréquence de 30 images par seconde, les images du corpus d'évaluation et du corpus d'entraînement sont très similaires. Ont se retrouve alors avec des courbes de loss de validation et d'entraînement comme celle-ci (pour une MSE) : ![No-PSNR](/images/training_curves/loss_curves_250ep_Adam_MSE_1.4.jpg "No PSNR")

Une piste qui a été explorée a été d'utiliser des vidéos de stop motion, celles-ci offrant une plus grande varation entre chaques images que de la vidéo filmée. L'effet sur la différentiation entre les courbes reste très modeste. Néanmoins cela permet d'introduire le choix des métriques utilisées.
La première intuition a été d'utiliser une Mean Square Error afin de comparer pixel par pixel (pour éviter les translations trop abrupts d'un objet par exemple). Cela donne des résultats d'apprentissages très convenables néanmoins la SSIM (Structural Similarity Index Measure) qui donne la similarité entre deux images dans leur construction et le PSNR (Peak Signal Noise Ratio) qui mesure la qualité de reconstruction d'une image ont également étaient testés pour l'apprentissage et gardés dans l'évaluation du modèle. 

Un exemple d'apprentissage :
Avec l'utilisation du PSNR comme loss ![PSNR descent](/images/training_curves/loss_curves_50ep_Adam_PSNR_1.1_BS7.jpg "PSNR descent")

Avec l'utilisation de la SSIM comme loss ![SSIM descent](/images/training_curves/loss_curves_50ep_Adam_SSIM_1.1_BS7.jpg "SSIM descent")

### Evaluation

Le modèle final produit un résultat plutôt satisfaisant au visionnage des vidéos auxquels on a augmenté le nombre d'images. Néanmoins, les vidéos contenant ayant à l'origine une fréquence basse (e.g., stop motion) contiennent encore des images où la fusion d'une image sur l'autre se fait encore ressentir (par exemple par l'apparation d'un objet en transparence). On notera tout de même que le modèle a besoin d'un certains nombre d'epochs d'apprentissage pour bien 'rendre' les couleurs et il a fallu appliquer un léger traitement post-modèle sur des problématique de luminosité car le modèle a encore tendance a sortir des images très légèrement trop sombre. 

Bien que son utilisation ai un champ limité, il est quand même intéressant de voir que le modèle arrive à ces performances avec une problématique de réduction importante de sa taille afin d'être utilisable sur une machine ayant des performances plutôt faible. Néanmoins, il serait intéressant d'essayer la mise en place d'un modèle capable de généraliser sur plusieurs vidéos afin d'éviter le temps d'apprentissage à chaque éxécutions. Une piste en ce sens serait d'utiliser une architecture à base de RNN ou de transformeurs qui sont plus efficaces lorsqu'il faut traiter des données qui ont une dimension temporelle. 







### Historique du projet (A compléter)
#### 04/12/2023 - Git creation

On this day I have created the git repository to upload my first 'naive' tries. I had not used Git for 1 year so it was quite troublesome. This files were push on the 'naive' branch. It contained the files 'Freq_inc.py', 'open_images.py' and 'slice_vid.py'. The first one contains a naive model of convolution that aims to generate an image given 2 images from a video that was sliced by the 2 other files (slice_vid.py had better performances so we only used slice_vid).
 