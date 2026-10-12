"""Numerical checks for the manuscript's feedback and regularization formulas."""

import math
import unittest

import torch

from trainers.ErrorCompass.losses import (
    confusion_adjacency,
    cpmc_loss,
    kl_teacher_loss,
    sccm_loss,
    update_confusion_counts,
)


class ConfusionTests(unittest.TestCase):
    def test_normalize_correct_counts_before_removing_diagonal(self):
        counts = torch.tensor([[8., 2., 0.], [1., 1., 0.], [0., 0., 0.]],
                              requires_grad=True)
        expected = torch.tensor([[0., .35, 0.], [.35, 0., 0.], [0., 0., 0.]])
        adjacency = confusion_adjacency(counts)
        torch.testing.assert_close(adjacency, expected)
        self.assertFalse(adjacency.requires_grad)
        self.assertEqual(counts[0, 0].item(), 8.)

    def test_empty_rows_have_no_invented_edges(self):
        for n_classes in (1, 3):
            with self.subTest(n_classes=n_classes):
                actual = confusion_adjacency(torch.zeros(n_classes, n_classes))
                torch.testing.assert_close(actual, torch.zeros_like(actual))
                self.assertTrue(torch.isfinite(actual).all())

    def test_empty_row_can_receive_a_real_reverse_edge(self):
        actual = confusion_adjacency(torch.tensor([[0., 2.], [0., 0.]]))
        torch.testing.assert_close(actual, torch.tensor([[0., .5], [.5, 0.]]))

    def test_ema_hard_counts_and_row_orientation(self):
        old = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
        logits = torch.tensor([[3., 1.], [1., 2.], [0., 8.], [4., 3.]],
                              requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1])
        actual = update_confusion_counts(old, logits, labels, rho=.75)
        expected = torch.tensor([[1.75, .5], [.25, 1.5]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(old, torch.tensor([[2., 0.], [0., 2.]]))
        torch.testing.assert_close(confusion_adjacency(actual),
                                   torch.tensor([[0., 23 / 126], [23 / 126, 0.]]))
        self.assertFalse(actual.requires_grad)

    def test_empty_batch_decays_existing_counts(self):
        actual = update_confusion_counts(torch.eye(2), torch.empty(0, 2),
                                         torch.empty(0, dtype=torch.long), .8)
        torch.testing.assert_close(actual, .8 * torch.eye(2))

    def test_single_class_predictions_are_counted_without_self_edge(self):
        actual = update_confusion_counts(torch.zeros(1, 1), torch.ones(3, 1),
                                         torch.zeros(3, dtype=torch.long), .5)
        torch.testing.assert_close(actual, torch.tensor([[1.5]]))
        torch.testing.assert_close(confusion_adjacency(actual), torch.zeros(1, 1))


class SCCMTests(unittest.TestCase):
    def test_class_mean_squared_distance_and_anchor_stop_gradient(self):
        prompt = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        anchors = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
        loss = sccm_loss(prompt, anchors)
        self.assertAlmostEqual(loss.item(), 2.)
        loss.backward()
        torch.testing.assert_close(prompt.grad, torch.tensor([[1., -1.], [-1., 1.]]))
        self.assertIsNone(anchors.grad)

    def test_single_class_is_not_averaged_over_dimensions(self):
        loss = sccm_loss(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]))
        self.assertAlmostEqual(loss.item(), 2.)


class CPMCTests(unittest.TestCase):
    def test_true_upper_triangle_topk_and_hand_computed_mean(self):
        prototypes = torch.tensor([[1., 0.], [.6, .8], [0., 1.]])
        # Large diagonal/lower entries must never compete with genuine pairs.
        confusion = torch.tensor([[99., .9, .8], [88., 99., .4], [77., 66., 99.]])
        loss = cpmc_loss(prototypes, confusion, top_k=2, threshold=.5,
                         margin_base=.2, margin_scale=1.)
        # Selected distances=.4 and 1; margins=1.1 and 1; (.9*.7 + .8*0)/2.
        self.assertAlmostEqual(loss.item(), .315, places=6)

    def test_threshold_is_strict_and_k_is_capped_at_real_pair_count(self):
        prototypes = torch.tensor([[1., 0.], [1., 0.]])
        confusion = torch.tensor([[0., .5], [.5, 0.]])
        self.assertEqual(cpmc_loss(prototypes, confusion, 10, .5, 1., 0.).item(), 0.)
        self.assertAlmostEqual(cpmc_loss(prototypes, confusion, 10, .49, 1., 0.).item(), .5)

    def test_empty_selected_pairs_return_differentiable_zero(self):
        cases = ((1, 3, .1), (3, 0, -.1), (3, 3, 0.))
        for n_classes, top_k, threshold in cases:
            with self.subTest(n_classes=n_classes, top_k=top_k, threshold=threshold):
                prototypes = torch.ones(n_classes, 2, requires_grad=True)
                loss = cpmc_loss(prototypes, torch.zeros(n_classes, n_classes),
                                 top_k, threshold, .5, .5)
                self.assertEqual(loss.item(), 0.)
                loss.backward()
                torch.testing.assert_close(prototypes.grad, torch.zeros_like(prototypes))

    def test_gradient_separates_confused_prototypes_and_detaches_graph(self):
        prototypes = torch.tensor([[1., 0.], [.8, .6]], requires_grad=True)
        confusion = torch.tensor([[0., .8], [.8, 0.]], requires_grad=True)
        loss = cpmc_loss(prototypes, confusion, 1, .1, 1., 0.)
        self.assertAlmostEqual(loss.item(), .64, places=6)
        loss.backward()
        updated = prototypes.detach() - .05 * prototypes.grad
        before = torch.nn.functional.cosine_similarity(prototypes[0], prototypes[1], dim=0)
        after = torch.nn.functional.cosine_similarity(updated[0], updated[1], dim=0)
        self.assertLess(after.item(), before.item())
        self.assertIsNone(confusion.grad)

    def test_zero_fp16_prototypes_remain_finite(self):
        prototypes = torch.zeros(2, 2, dtype=torch.float16)
        loss = cpmc_loss(prototypes, torch.ones(2, 2), 1, .1, .2, .5)
        self.assertTrue(torch.isfinite(loss))


class KLTests(unittest.TestCase):
    def test_teacher_to_student_kl_batch_mean_and_detach(self):
        student = torch.zeros(2, 2, requires_grad=True)
        teacher = torch.tensor([[.75, .25], [.25, .75]]).log().requires_grad_()
        loss = kl_teacher_loss(student, teacher)
        expected = .75 * math.log(1.5) + .25 * math.log(.5)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.tensor([[-.125, .125], [.125, -.125]]))
        self.assertIsNone(teacher.grad)

    def test_single_class_kl_is_zero_and_differentiable(self):
        student = torch.tensor([[2.], [4.]], requires_grad=True)
        loss = kl_teacher_loss(student, torch.tensor([[1.], [5.]]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))


if __name__ == "__main__":
    unittest.main()
