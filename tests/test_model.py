"""Tests de forme de la baseline 2.5D.

Ces tests utilisent des tenseurs artificiels. Ils vérifient le contrat du
modèle sans dépendre du chargeur de données développé sur l'autre branche.
"""

import unittest

import torch

from src.models import TinyUNet25D, count_trainable_parameters


class TinyUNet25DTests(unittest.TestCase):
    """Vérifie les propriétés minimales avant tout entraînement."""

    def setUp(self) -> None:
        # Une nouvelle instance par test évite qu'un test modifie l'état d'un
        # autre test et rende les résultats difficiles à interpréter.
        self.model = TinyUNet25D()
        self.model.eval()

    def assert_output_shape(self, height: int, width: int) -> None:
        """Vérifie que le modèle conserve la résolution spatiale."""

        fake_mri_slices = torch.randn(2, 6, height, width)
        with torch.no_grad():
            logits = self.model(fake_mri_slices)
        self.assertEqual(logits.shape, (2, 4, height, width))

    def test_standard_subject_shape(self) -> None:
        """Les sujets habituels donnent des coupes 144 x 192."""

        self.assert_output_shape(height=144, width=192)

    def test_variable_subject_shape(self) -> None:
        """Le sujet 23 peut produire des coupes 160 x 192."""

        self.assert_output_shape(height=160, width=192)

    def test_wrong_number_of_channels_is_rejected(self) -> None:
        """Une erreur explicite vaut mieux qu'un échec obscur plus loin."""

        invalid_input = torch.randn(1, 5, 144, 192)
        with self.assertRaisesRegex(ValueError, "attend 6 canaux"):
            self.model(invalid_input)

    def test_parameter_budget(self) -> None:
        """La baseline doit rester sous notre budget interne de 100k poids."""

        parameter_count = count_trainable_parameters(self.model)
        self.assertLess(parameter_count, 100_000)


if __name__ == "__main__":
    unittest.main()
