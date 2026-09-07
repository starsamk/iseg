"""Baseline U-Net 2.5D légère pour segmenter une coupe iSeg.

Le terme « 2.5D » décrit uniquement la préparation de l'entrée : le réseau
reçoit plusieurs coupes 2D voisines comme des canaux, mais toutes ses
convolutions restent 2D. Avec T1 et T2 aux positions z-1, z et z+1, une entrée
contient donc six canaux.

Le réseau produit quatre scores par pixel : fond, LCR, substance grise et
substance blanche. Ces scores bruts sont appelés « logits ». La fonction de
perte les convertira implicitement en probabilités pendant l'entraînement.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Deux convolutions 3x3 qui apprennent des motifs locaux.

    Une convolution 3x3 regarde chaque pixel et ses voisins immédiats. Deux
    convolutions successives combinent progressivement des informations sur
    une zone plus large. ReLU introduit une non-linéarité : sans elle, empiler
    des convolutions resterait une transformation linéaire.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    """Agrandit une représentation puis récupère les détails de l'encodeur."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolutions = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Utiliser la taille exacte du skip rend le modèle robuste même si H ou W
        # ne sont pas parfaitement divisibles par huit.
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Le skip réinjecte les contours fins conservés par l'encodeur.
        x = torch.cat((x, skip), dim=1)
        return self.convolutions(x)


class TinyUNet25D(nn.Module):
    """U-Net 2D compacte recevant six coupes IRM empilées.

    Forme d'entrée : ``[batch, 6, hauteur, largeur]``.
    Forme de sortie : ``[batch, 4, hauteur, largeur]``.

    Les canaux suivent 6 -> 12 -> 24 -> 48 dans l'encodeur. À chaque
    MaxPool2d, hauteur et largeur sont divisées par deux tandis que le nombre de
    canaux augmente. Le décodeur effectue ensuite le chemin inverse.
    """

    def __init__(
        self,
        input_channels: int = 6,
        number_of_classes: int = 4,
        base_channels: int = 6,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels doit être strictement positif")
        if number_of_classes <= 1:
            raise ValueError("number_of_classes doit être supérieur à un")
        if base_channels <= 0:
            raise ValueError("base_channels doit être strictement positif")

        self.input_channels = input_channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Encodeur : chaque niveau extrait des caractéristiques plus abstraites.
        self.encoder_level_1 = DoubleConv(input_channels, c1)
        self.encoder_level_2 = DoubleConv(c1, c2)
        self.encoder_level_3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Décodeur : il remonte vers la résolution de la coupe d'origine.
        self.decoder_level_3 = DecoderBlock(c4, c3, c3)
        self.decoder_level_2 = DecoderBlock(c3, c2, c2)
        self.decoder_level_1 = DecoderBlock(c2, c1, c1)

        # Une convolution 1x1 produit un score par classe pour chaque pixel.
        self.classifier = nn.Conv2d(c1, number_of_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "L'entrée doit avoir la forme [batch, canaux, hauteur, largeur], "
                f"forme reçue : {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Le modèle attend {self.input_channels} canaux, "
                f"mais en a reçu {x.shape[1]}"
            )

        # e1, e2 et e3 sont conservés pour les connexions U-Net.
        e1 = self.encoder_level_1(x)
        e2 = self.encoder_level_2(self.pool(e1))
        e3 = self.encoder_level_3(self.pool(e2))
        latent = self.bottleneck(self.pool(e3))

        d3 = self.decoder_level_3(latent, e3)
        d2 = self.decoder_level_2(d3, e2)
        d1 = self.decoder_level_1(d2, e1)
        return self.classifier(d1)


def count_trainable_parameters(model: nn.Module) -> int:
    """Compte uniquement les valeurs que l'optimiseur pourra modifier."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
