import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DistillationTest(unittest.TestCase):
    def test_layer_mapping_4_to_12(self):
        from train_tinybert import make_layer_pairs

        pairs = make_layer_pairs(4, 12)
        self.assertEqual([pair.teacher_hidden for pair in pairs], [3, 6, 9, 12])
        self.assertEqual([pair.teacher_attention for pair in pairs], [2, 5, 8, 11])

    def test_prediction_loss_is_near_zero_for_equal_logits(self):
        from train_tinybert import prediction_distillation_loss

        logits = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        loss = prediction_distillation_loss(logits, logits, temperature=4.0)
        self.assertLess(abs(float(loss)), 1e-6)

    def test_masked_hidden_loss_ignores_padding(self):
        from train_tinybert import LayerPair, hidden_distillation_loss

        projection = torch.nn.Linear(2, 2, bias=False)
        projection.weight.data.copy_(torch.eye(2))
        student = (torch.zeros(1, 2, 2), torch.tensor([[[1.0, 2.0], [99.0, 99.0]]]))
        teacher = (torch.zeros(1, 2, 2), torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]))
        loss = hidden_distillation_loss(
            student,
            teacher,
            torch.nn.ModuleList([projection]),
            [LayerPair(1, 1, 0, 0)],
            torch.tensor([[1, 0]]),
        )
        self.assertEqual(float(loss), 0.0)

    def test_attention_loss_supports_different_head_counts(self):
        from train_tinybert import LayerPair, attention_distillation_loss

        # Both head-averaged maps are identical on the only valid token. Large
        # differences in the padded row/column must not contribute.
        student_map = torch.tensor(
            [[[[1.0, 99.0], [99.0, 99.0]], [[1.0, 50.0], [50.0, 50.0]]]]
        )
        teacher_map = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        loss = attention_distillation_loss(
            (student_map,),
            (teacher_map,),
            [LayerPair(1, 1, 0, 0)],
            torch.tensor([[1, 0]]),
        )
        self.assertEqual(float(loss), 0.0)

    def test_attention_loss_is_headwise_when_counts_match(self):
        from train_tinybert import LayerPair, attention_distillation_loss

        # The two models have identical head averages but swapped individual
        # heads. Direct head-wise comparison must therefore produce a loss.
        student_map = torch.tensor([[[[1.0]], [[0.0]]]])
        teacher_map = torch.tensor([[[[0.0]], [[1.0]]]])
        loss = attention_distillation_loss(
            (student_map,),
            (teacher_map,),
            [LayerPair(1, 1, 0, 0)],
            torch.tensor([[1]]),
        )
        self.assertEqual(float(loss), 1.0)


if __name__ == "__main__":
    unittest.main()
