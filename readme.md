# Deep Learning & Differentiable Programming Mini-project
## VFI (Video Frame Interpolation)

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

L'architecture d'origine de U-Net est la suivante ![U-Net architecture](/images/u-net-architecture-1024x682.png "U-Net architecture")



### Historique du projet
#### 04/12/2023 - Git creation

On this day I have created the git repository to upload my first 'naive' tries. I had not used Git for 1 year so it was quite troublesome. This files were push on the 'naive' branch. It contained the files 'Freq_inc.py', 'open_images.py' and 'slice_vid.py'. The first one contains a naive model of convolution that aims to generate an image given 2 images from a video that was sliced by the 2 other files (slice_vid.py had better performances so we only used slice_vid).
