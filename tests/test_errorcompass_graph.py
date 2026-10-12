"""Small, hand-computable checks for the manuscript's SCGP equations."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


# Import only the pure torch module; these tests do not require Dassl,
# transformers, dataset files, or pretrained model weights.
MODULE_PATH = Path(__file__).resolve().parents[1] / "trainers" / "ErrorCompass" / "graph.py"
SPEC = importlib.util.spec_from_file_location("errorcompass_graph", MODULE_PATH)
graph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(graph)


class SemanticAdjacencyTests(unittest.TestCase):
    def test_hand_calculated_cosines_and_single_semantic_diagonal(self):
        anchors = torch.tensor([[5.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
        # Normalized cosine matrix [[1, .6], [.6, 1]], row sums 1.6.
        expected = torch.tensor([[0.625, 0.375], [0.375, 0.625]], dtype=torch.float64)
        torch.testing.assert_close(graph.semantic_adjacency(anchors, 0.5), expected)

    def test_one_random_threshold_is_shared_by_all_rows(self):
        with patch.object(graph.random, "uniform", return_value=0.55) as sample:
            result = graph.semantic_adjacency(torch.eye(3))
        sample.assert_called_once_with(0.5, 0.6)
        torch.testing.assert_close(result, torch.eye(3))

    def test_negative_similarities_are_removed(self):
        anchors = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        torch.testing.assert_close(graph.semantic_adjacency(anchors, 0.5), torch.eye(3))

    def test_threshold_boundary_is_inclusive(self):
        anchors = torch.tensor([[1.0, 0.0], [0.5, 3.0 ** 0.5 / 2.0]], dtype=torch.float64)
        expected = torch.tensor([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]], dtype=torch.float64)
        torch.testing.assert_close(graph.semantic_adjacency(anchors, 0.5), expected)

    def test_zero_anchor_and_half_precision_do_not_create_nan(self):
        for dtype in (torch.float32, torch.float16):
            anchors = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
            expected = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
            torch.testing.assert_close(graph.semantic_adjacency(anchors, 0.5), expected)

    def test_empty_malformed_and_nonfinite_inputs_fail(self):
        for anchors in (torch.empty(0, 2), torch.empty(2, 0), torch.ones(2),
                        torch.ones(1, 2, 2), torch.tensor([[float("nan")]])):
            with self.subTest(shape=anchors.shape), self.assertRaises(ValueError):
                graph.semantic_adjacency(anchors)
        with self.assertRaises(TypeError):
            graph.semantic_adjacency(torch.ones(2, 2, dtype=torch.int64))
        for threshold in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                graph.semantic_adjacency(torch.eye(2), threshold)

    def test_gradient_through_retained_similarities(self):
        anchors = torch.tensor([[1.0, 0.0], [0.8, 0.6]], dtype=torch.float64, requires_grad=True)
        graph.semantic_adjacency(anchors, 0.5)[0, 1].backward()
        self.assertTrue(torch.isfinite(anchors.grad).all())
        self.assertGreater(anchors.grad.abs().sum().item(), 0.0)


class ContextAdjacencyTests(unittest.TestCase):
    def test_confusion_magnitude_is_preserved(self):
        semantic = torch.eye(2)
        confusion = torch.tensor([[0.0, 0.2], [0.2, 0.0]])
        expected = torch.tensor([[0.3, 0.14], [0.14, 0.3]])
        result = graph.context_adjacency(semantic, confusion, 0.3, True)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.sum(1), torch.tensor([0.44, 0.44]))

    def test_warmup_and_transfer_return_semantics_without_confusion(self):
        semantic = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        self.assertIs(graph.context_adjacency(semantic, None, 0.3, False), semantic)

    def test_zero_confusion_after_activation_keeps_semantic_coefficient(self):
        result = graph.context_adjacency(torch.eye(2), torch.zeros(2, 2), 0.3, True)
        torch.testing.assert_close(result, 0.3 * torch.eye(2))

    def test_invalid_shapes_and_weights_fail(self):
        for semantic, confusion in ((torch.ones(2, 3), torch.eye(2)),
                                    (torch.eye(2), torch.eye(3)),
                                    (torch.empty(0, 0), torch.empty(0, 0))):
            with self.subTest(shape=semantic.shape), self.assertRaises(ValueError):
                graph.context_adjacency(semantic, confusion, 0.3, True)
        for weight in (-0.1, 1.1, float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                graph.context_adjacency(torch.eye(2), torch.eye(2), weight, True)


class GraphConvolutionTests(unittest.TestCase):
    def identity_layer(self, dim=2):
        layer = graph.GraphConvolution(dim).double()
        with torch.no_grad():
            layer.propagation_weight.copy_(torch.eye(dim, dtype=torch.float64))
        return layer

    def test_only_shared_weight_is_learned(self):
        layer = self.identity_layer()
        self.assertEqual(list(dict(layer.named_parameters())), ["propagation_weight"])
        self.assertEqual(tuple(layer.propagation_weight.shape), (2, 2))

    def test_one_self_loop_addition_with_existing_diagonal(self):
        layer = self.identity_layer()
        # A+I = [[1.5,.5],[.5,1.5]], degree 2; a second I would
        # incorrectly produce diagonal 5/6 instead of 3/4.
        adjacency = torch.full((2, 2), 0.5, dtype=torch.float64)
        expected = torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64)
        torch.testing.assert_close(layer(torch.eye(2, dtype=torch.float64), adjacency), expected)
        torch.testing.assert_close(adjacency, torch.full((2, 2), 0.5, dtype=torch.float64))

    def test_unequal_degrees_use_symmetric_not_row_normalization(self):
        layer = self.identity_layer()
        adjacency = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        # Degrees of A+I are [2,4]: diagonal [1/2,3/4], edge 1/sqrt(8).
        edge = 1.0 / (8.0 ** 0.5)
        expected = torch.tensor([[0.5, edge], [edge, 0.75]], dtype=torch.float64)
        torch.testing.assert_close(layer(torch.eye(2, dtype=torch.float64), adjacency), expected)

    def test_other_class_prototype_changes_this_class_output(self):
        layer = self.identity_layer()
        adjacency = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        features = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
        torch.testing.assert_close(layer(features, adjacency),
                                   torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64))
        features[1, 1] = 8.0
        torch.testing.assert_close(layer(features, adjacency)[0],
                                   torch.tensor([1.0, 4.0], dtype=torch.float64))

    def test_leaky_relu_slope_and_all_classes_are_retained(self):
        layer = self.identity_layer()
        features = torch.tensor([[-2.0, 3.0], [4.0, -5.0], [-1.0, -1.0]], dtype=torch.float64)
        expected = torch.tensor([[-0.4, 3.0], [4.0, -1.0], [-0.2, -0.2]], dtype=torch.float64)
        torch.testing.assert_close(layer(features, torch.zeros(3, 3, dtype=torch.float64)), expected)

    def test_gradients_reach_features_adjacency_and_weight(self):
        layer = self.identity_layer()
        features = torch.tensor([[2.0, -1.0], [1.0, 4.0]], dtype=torch.float64, requires_grad=True)
        adjacency = torch.tensor([[0.2, 0.3], [0.1, 0.4]], dtype=torch.float64, requires_grad=True)
        layer(features, adjacency).square().sum().backward()
        for tensor in (features, adjacency, layer.propagation_weight):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

    def test_multiple_layers_keep_class_dimension(self):
        first, second = self.identity_layer(), self.identity_layer()
        features = torch.tensor([[-5.0, 2.0]], dtype=torch.float64)
        adjacency = torch.zeros(1, 1, dtype=torch.float64)
        # Two separate LeakyReLU applications, one in each graph layer.
        torch.testing.assert_close(second(first(features, adjacency), adjacency),
                                   torch.tensor([[-0.2, 2.0]], dtype=torch.float64))

    def test_empty_malformed_and_negative_adjacency_fail(self):
        layer = self.identity_layer()
        for features, adjacency in ((torch.empty(0, 2), torch.empty(0, 0)),
                                    (torch.ones(2, 3), torch.eye(2)),
                                    (torch.ones(2, 2), torch.eye(3)),
                                    (torch.ones(2, 2), torch.ones(2, 3)),
                                    (torch.ones(2, 2), -torch.eye(2))):
            with self.subTest(features=features.shape, adjacency=adjacency.shape), self.assertRaises(ValueError):
                layer(features, adjacency)
        for dim in (0, -1, 2.5, True):
            with self.subTest(dim=dim), self.assertRaises(ValueError):
                graph.GraphConvolution(dim)


if __name__ == "__main__":
    unittest.main()
