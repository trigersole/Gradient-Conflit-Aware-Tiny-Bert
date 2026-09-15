import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GradientConflictProbeTest(unittest.TestCase):
    def test_cosines_summary_and_no_grad_side_effect(self):
        from gradient_conflict import GradientConflictProbe

        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        with tempfile.TemporaryDirectory() as directory:
            probe = GradientConflictProbe(
                [parameter], every_n_steps=2, csv_path=Path(directory) / "cosines.csv"
            )
            losses = {
                "task": parameter[0] + parameter[1],       # [1, 1]
                "prediction": 2 * parameter.sum(),        # [2, 2], cosine 1
                "hidden": -parameter.sum(),               # [-1, -1], cosine -1
                "attention": parameter[0] - parameter[1], # [1, -1], cosine 0
            }

            self.assertIsNone(probe.measure(losses, step=1))
            row = probe.measure(losses, step=2, epoch=0)
            self.assertAlmostEqual(row["cos_task_prediction"], 1.0)
            self.assertAlmostEqual(row["cos_task_hidden"], -1.0)
            self.assertAlmostEqual(row["cos_task_attention"], 0.0)
            self.assertIsNone(parameter.grad)
            self.assertEqual(probe.summary()["hidden"]["negative_percent"], 100.0)
            probe.close()
            self.assertTrue((Path(directory) / "cosines.csv").exists())

    def test_zero_norm_is_excluded(self):
        from gradient_conflict import GradientConflictProbe

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        probe = GradientConflictProbe([parameter], every_n_steps=1)
        losses = {
            "task": parameter.square().sum(),
            "prediction": (parameter * 0).sum(),
            "hidden": parameter.sum(),
            "attention": parameter.sum(),
        }
        row = probe.measure(losses, step=1)
        self.assertTrue(math.isnan(row["cos_task_prediction"]))
        self.assertEqual(probe.summary()["prediction"]["measured_batches"], 0)


if __name__ == "__main__":
    unittest.main()
