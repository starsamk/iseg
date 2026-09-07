"""Modèles utilisés par le projet iSeg."""

from .tiny_unet_2p5d import TinyUNet25D, count_trainable_parameters

__all__ = ["TinyUNet25D", "count_trainable_parameters"]
