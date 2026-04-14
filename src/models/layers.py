import torch.nn as nn  # Import de la bibliothèque de modules de PyTorch pour construire le réseau de neurones

class ConvBlock(nn.Module): # Definition d'un bloc de convolution qui hérite de nn.Module
        """"Bloc de convolution comprenant une convolution, une activation ReLU et un max pooling
        Args:
            in_channels (int): Nombre de canaux d'entrée pour la convolution
            out_channels (int): Nombre de filtres de sortie pour la convolution
        """
        def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,pool= True):
            super(ConvBlock,self).__init__()  # Appel du constructeur de la classe parente nn.Module
            layers = [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding),  # Convolution 2D avec les paramètres spécifiés
                      nn.ReLU()]  # Fonction d'activation ReLU
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2))  # Ajouter le max pooling si spécifié
            self.block = nn.Sequential(*layers)  # Créer un module séquentiel avec les couches définies

        def forward(self, x):  # Définition de la méthode forward pour la propagation avant du bloc
            return self.block(x)  # Retourner les caractéristiques extraites par le bloc