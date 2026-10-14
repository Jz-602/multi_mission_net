"""CPU regressions for the finite-gradient guard; execute this file directly."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.trainer.three_task_runtime import CheckFiniteGradients


class FiniteGradientsTest(unittest.TestCase):
    @staticmethod
    def BuildModel(count=3):
        model = torch.nn.Module()
        for index in range(count):
            model.register_parameter(f"parameter_{index}", torch.nn.Parameter(torch.ones(4)))
        return model

    def TestFiniteAndMissingGradients(self):
        model = self.BuildModel()
        model.parameter_0.grad = torch.arange(4, dtype=torch.float32)
        model.parameter_2.grad = torch.full((4,), -3.0)
        original_gradients = [parameter.grad for parameter in model.parameters()]
        CheckFiniteGradients(model)
        for parameter, original_gradient in zip(model.parameters(), original_gradients):
            self.assertIs(parameter.grad, original_gradient)

    def TestNoGradientsAndNoParameters(self):
        CheckFiniteGradients(self.BuildModel())
        CheckFiniteGradients(torch.nn.Module())

    def TestNonfiniteGradientsReportTheirNames(self):
        for nonfinite_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=nonfinite_value):
                model = self.BuildModel()
                model.parameter_0.grad = torch.ones(4)
                model.parameter_2.grad = torch.tensor([0.0, nonfinite_value, 1.0, 2.0])
                with self.assertRaises(FloatingPointError) as caught:
                    CheckFiniteGradients(model)
                self.assertEqual(
                    str(caught.exception),
                    "non-finite gradients detected before optimizer.step: parameter_2",
                )

    def TestFirstTenNonfiniteNamesPreserveParameterOrder(self):
        model = self.BuildModel(13)
        for parameter in model.parameters():
            parameter.grad = torch.full((4,), float("nan"))
        with self.assertRaises(FloatingPointError) as caught:
            CheckFiniteGradients(model)
        self.assertEqual(
            str(caught.exception),
            "non-finite gradients detected before optimizer.step: "
            + ", ".join(f"parameter_{index}" for index in range(10)),
        )


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.testMethodPrefix = "Test"
    unittest.main(testLoader=loader)
