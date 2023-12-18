# Deep Learning & Differentiable Programming Mini-project
## VFI (Video Frame Interpolation)

### Contexte

L'objectif de ce projet est de créer un modèle basé sur un ou une combinaison de réseaux de neurones (MLP, LSTM, ConvNet, Transformer...), de l'entrainer sur une base de données, évaluer et analyser ses performances puis itérer pour l'améliorer dans sa tâche ou le remplacer par un autre modèle.

La tâche qui a été choisie ici est celle de l'interpolation d'image en vidéo ou Video Frame Interpolation (VFI). Celle-ci consiste à déterminer une nouvelle image entre 2 images adjacentes d'une vidéo. Parmi les nombreuses applications on peut notamment citer l'augmentation du nombre de frame (dans la restauration de vieilles vidéos ou pour fluidifier des vidéos en stop motion par exemple), les ralentis de vidéo ou encore la fluidification du flux vidéo (télévision, streaming).

### Travaux actuels

Une méthode de VFI couramment utilisée aujourd'hui est la recherche du 'flux optique' des objets présents sur l'image. L'idée générale est que l'on cherche à déterminer le mouvement d'un objet (groupe de pixels) entre plusieurs images successives afin de le déplacer à une localisation intermédiaire. Les algorithmes généralement utilisés pour identifier et applique le flux optique dans des systèmes comme des écrans de télévision sont déterministes, cependant depuis 2017 des algorithmes de réseaux neuronaux profonds ont été développés afin de déterminer le flux optique. 

Une autre approche appelée "flow-free" consiste à appliquer des convolutions et fusionner les images adjacentes. Nous allons ici nous inspirer de ces approches pour notre modèle.

### Historique du projet
#### 04/12/2023 - Git creation

On this day I have created the git repository to upload my first 'naive' tries. I had not used Git for 1 year so it was quite troublesome. This files were push on the 'naive' branch. It contained the files 'Freq_inc.py', 'open_images.py' and 'slice_vid.py'. The first one contains a naive model of convolution that aims to generate an image given 2 images from a video that was sliced by the 2 other files (slice_vid.py had better performances so we only used slice_vid).
