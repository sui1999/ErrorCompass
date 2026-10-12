"""Behavioral invariants implied by the current TMM method and protocol."""

import contextlib
import copy
import io
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_errorcompass_integration import (
    implementation, configuration, TinyBackbone, NAMES, TEMPLATES,
    IMAGES, LABELS, make_model,
)
from trainers.ErrorCompass.losses import chc_loss, cpmc_loss


class CHCContractTests(unittest.TestCase):
    def test_binary_is_cross_entropy_in_value_and_gradient_even_with_feedback(self):
        logits = torch.tensor([[80., -30.], [-20., 60.], [0.2, 0.7]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([1, 0, 1])
        graph = torch.tensor([[0., .9], [.9, 0.]], requires_grad=True)
        actual = chc_loss(logits, labels, 9, 5., graph, .25)
        expected = F.cross_entropy(logits, labels)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits)[0])
        self.assertIsNone(graph.grad)

    def test_multiclass_matches_direct_formula_in_value_and_weight_gradient(self):
        logits = torch.tensor([[.2, .5, -.1, .7]], dtype=torch.float64, requires_grad=True)
        graph = torch.tensor([[0., .4, .1, .8], [.4, 0., 0., 0.],
                              [.1, 0., 0., 0.], [.8, 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        weights = torch.exp(2. * logits[0, [3, 1]])
        weights = weights * torch.tensor([1.2, 1.1], dtype=torch.float64)
        weights = weights / weights.sum()
        expected = torch.log(torch.exp(logits[0, 0]) + (weights * logits[0, [3, 1]].exp()).sum()) - logits[0, 0]
        actual = chc_loss(logits, torch.tensor([0]), 2, 2., graph, .25)
        torch.testing.assert_close(actual, expected)
        grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.autograd.grad(expected, logits)[0])
        self.assertEqual(grad[0, 2].item(), 0.)
        self.assertIsNone(graph.grad)

    def test_saturation_is_identical_for_both_selection_parameters(self):
        logits = torch.tensor([[.1, .5, .3, -.1], [.4, .2, -.7, .1]])
        labels = torch.tensor([0, 2])
        graph = torch.tensor([[0., .6, .4, .7], [.6, 0., .3, .2], [.4, .3, 0., .5], [.7, .2, .5, 0.]])
        reference = chc_loss(logits, labels, 3, 5., graph)
        for k in [5, 7, 9]:
            torch.testing.assert_close(chc_loss(logits, labels, k, 5., graph), reference)
        prototypes = torch.tensor([[1., 0.], [.8, .2], [.6, .4], [.3, .7]])
        torch.testing.assert_close(cpmc_loss(prototypes, graph, 7, .2, .5, .5),
                                   cpmc_loss(prototypes, graph, 10, .2, .5, .5))

    def test_extreme_logits_remain_finite_with_shift_invariance(self):
        logits = torch.tensor([[-90., 100., 99.]], dtype=torch.float64, requires_grad=True)
        loss = chc_loss(logits, torch.tensor([0]), 2, 5.)
        torch.testing.assert_close(loss, chc_loss(logits+10000., torch.tensor([0]), 2, 5.))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class ModelContractTests(unittest.TestCase):
    def test_fusion_preserves_branch_scales_and_clips_only_shared_scalar(self):
        model = make_model()
        learner = model.prompt_learner
        t = F.normalize(torch.tensor([[1., 1.], [1., -1.]]), dim=1)
        anchors = torch.tensor([[1., 0.], [0., 1.]])
        visual = torch.tensor([[.1, .2], [.5, -.1]])
        propagated = torch.tensor([[2., -3.], [4., .5]])
        for value in [-1., .4, 2.]:
            learner.scale_mm.data.fill_(value)
            with patch.object(learner, 'propagate_visual_prototypes', return_value=propagated):
                actual, p = learner.apply_scgp(t, anchors, visual)
            r = .3*anchors+.7*(.5*visual+.5*propagated)
            expected = F.normalize(t+min(max(value, 0.), 1.5)*(r-t), dim=1)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(p, propagated)

    def test_default_trainables_and_fixed_nonunit_visual_means(self):
        model = make_model(SCCM_LAMBDA=1., KL_LAMBDA=1.)
        trainable = {n for n,p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {'prompt_learner.ctx', 'prompt_learner.scale_mm',
                                    'prompt_learner.scgp_propagation.propagation_weight'})
        self.assertEqual(model.prompt_learner.scale_mm.item(), 1.)
        self.assertLessEqual(model.prompt_learner.scgp_propagation.propagation_weight.abs().max().item(), .5)
        expected = F.normalize(IMAGES,dim=1)[LABELS==0].mean(0)
        torch.testing.assert_close(model.base_img_features[0], expected)
        self.assertLess(model.base_img_features[0].norm().item(), .99)

    def test_feedback_refresh_affects_current_losses_and_next_graph_forward(self):
        model=make_model()
        model.current_epoch=5
        learner=model.prompt_learner
        learner.chc_step_counter.fill_(9)
        learner.confusion_matrix[0,1]=1.
        seen=[]
        original=learner.build_context_adjacency
        def capture(semantic, current_epoch=0):
            seen.append(learner.confusion_adjacency.clone())
            return original(semantic,current_epoch)
        with patch.object(learner, 'build_context_adjacency', side_effect=capture), \
             patch.object(learner, 'compute_chc_loss', wraps=learner.compute_chc_loss) as chc:
            result=model(IMAGES,LABELS)
            current=learner.confusion_adjacency.clone()
            self.assertEqual(learner.chc_step_counter.item(),10)
            self.assertEqual(seen[0].sum().item(),0.)
            self.assertGreater(current.sum().item(),0.)
            expected=chc_loss(result[0],LABELS,3,5.,current,.25)*.5
            torch.testing.assert_close(result[4],expected)
            model(IMAGES,LABELS)
            torch.testing.assert_close(seen[1],current)
            self.assertEqual(chc.call_count,2)

    def test_novel_never_reads_support_or_uses_confusion_even_at_late_epoch(self):
        class ForbiddenLoader:
            def __iter__(self): raise AssertionError('Novel support must not be read')
        cfg=configuration(); cfg.DATASET.SUBSAMPLE_CLASSES='new'
        with patch.object(implementation,'BIOMEDCOOP_TEMPLATES',TEMPLATES):
            model=implementation.ErrorCompassModel(cfg,NAMES,TinyBackbone(),ForbiddenLoader()).eval()
        model.current_epoch=99
        model.prompt_learner.confusion_adjacency.fill_(1.)
        torch.testing.assert_close(model.base_img_features,model.base_text_features)
        semantic=torch.eye(3)
        torch.testing.assert_close(model.prompt_learner.build_context_adjacency(semantic,99),semantic)
        before=model.prompt_learner.confusion_matrix.clone()
        model(IMAGES)
        torch.testing.assert_close(model.prompt_learner.confusion_matrix,before)

    def test_actual_transfer_uses_target_images_and_preserves_source_shot_coefficients(self):
        source=make_model().eval()
        trainer=implementation.ErrorCompass_BiomedCLIP()
        trainer.model=source; trainer.device=torch.device('cpu')
        cfg=copy.deepcopy(source.cfg); cfg.DATASET.SUBSAMPLE_CLASSES='all'; cfg.DATASET.NUM_SHOTS=16
        support=[{'img':IMAGES.flip(1), 'label':LABELS}]
        before={k:v.clone() for k,v in source.state_dict().items()}
        with patch.object(implementation,'BIOMEDCOOP_TEMPLATES',TEMPLATES), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            target=trainer.build_target_model(cfg,NAMES,support)
        expected=F.normalize(IMAGES.flip(1),dim=1)[LABELS==0].mean(0)
        torch.testing.assert_close(target.base_img_features[0],expected)
        self.assertFalse(torch.allclose(target.base_img_features,target.base_text_features))
        torch.testing.assert_close(target.prompt_learner.alpha_it,source.prompt_learner.alpha_it)
        self.assertAlmostEqual(target.prompt_learner.alpha_it.item(),.5)
        self.assertFalse(target.prompt_learner.feedback_enabled)
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))
        target(IMAGES)
        for k,v in source.state_dict().items(): torch.testing.assert_close(v,before[k])

    def test_optional_gate_and_temperature_are_preserved(self):
        model=make_model(USE_SCGP_ADAPTIVE_GATE=True,CHC_TAU=2.)
        outputs=model(IMAGES,LABELS)
        expected=chc_loss(outputs[0]/2.,LABELS,3,5.)*.5
        torch.testing.assert_close(outputs[4],expected)
        sum(outputs[1:]).backward()
        self.assertIsNotNone(model.prompt_learner.scgp_alpha_scale.grad)
        self.assertTrue(torch.isfinite(model.prompt_learner.scgp_alpha_scale.grad))


if __name__=='__main__':
    unittest.main()
