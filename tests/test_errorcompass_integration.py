"""Execute the real model with a tiny deterministic frozen backbone.

Only unavailable external framework imports and backbone/tokenization are
substituted. The actual model, prompt learner, graph, losses, feedback updates,
and checkpoint-version checks run unchanged on real CPU torch tensors.
"""

import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]


class _Registry:
    def register(self):
        return lambda cls: cls


class _ExternalTrainerBase:
    def model_backward_and_update(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def _module(name, **members):
    result = types.ModuleType(name)
    result.__dict__.update(members)
    result.__path__ = []
    return result


def _unavailable(*args, **kwargs):
    raise AssertionError("The integration model must not load external resources")


def _import_real_model():
    replacements = {
        "dassl": _module("dassl"),
        "dassl.engine": _module("dassl.engine", TRAINER_REGISTRY=_Registry(), TrainerX=_ExternalTrainerBase),
        "dassl.utils": _module("dassl.utils", load_checkpoint=_unavailable),
        "dassl.optim": _module("dassl.optim", build_optimizer=_unavailable, build_lr_scheduler=_unavailable),
        "dassl.metrics": _module("dassl.metrics", compute_accuracy=_unavailable),
        "open_clip": _module("open_clip"),
        "open_clip.src": _module("open_clip.src"),
        "open_clip.src.open_clip": _module("open_clip.src.open_clip"),
        "open_clip.src.open_clip.factory": _module(
            "open_clip.src.open_clip.factory", create_model_and_transforms=_unavailable,
            HF_HUB_PREFIX="hf-hub:", _MODEL_CONFIGS={}),
        "open_clip.src.open_clip.tokenizer": _module(
            "open_clip.src.open_clip.tokenizer", HFTokenizer=_unavailable),
    }
    with patch.dict(sys.modules, replacements), patch.object(sys, "path", [str(ROOT)] + sys.path):
        return importlib.import_module("trainers.ErrorCompass.errorcompass_biomedclip")


implementation = _import_real_model()
_accuracy_spec = importlib.util.spec_from_file_location(
    "integration_dassl_accuracy", ROOT / "Dassl.pytorch" / "dassl" / "metrics" / "accuracy.py")
_accuracy_module = importlib.util.module_from_spec(_accuracy_spec)
_accuracy_spec.loader.exec_module(_accuracy_module)


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    __setattr__ = dict.__setitem__


def configuration(**overrides):
    options = Config(
        N_CTX=4, CTX_INIT="a photo of a", CSC=False,
        CLASS_TOKEN_POSITION="end", N_PROMPTS=3, SCCM_LAMBDA=0.7,
        KL_LAMBDA=0.4, PREC="fp32", CHECKPOINT_DIR="unused",
    )
    options.update(overrides)
    return Config(
        TRAINER=Config(ERRORCOMPASS=options),
        DATASET=Config(NUM_SHOTS=4, SUBSAMPLE_CLASSES="base", NAME="tiny"),
        INPUT=Config(SIZE=[224, 224]),
        OPTIM=Config(MAX_EPOCH=100),
    )


class TinyTokenizer:
    """Fixed-length batch tokenizer, including the HF wrapper's encode API."""
    pad_token_id = 0
    sep_token_id = 2

    def __init__(self):
        self.tokenizer = self
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=True):
        ids = []
        for word in re.findall(r"[a-z]+|[.]", text.lower()):
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary) + 3
            ids.append(self.vocabulary[word])
        return [1] + ids + [2] if add_special_tokens else ids

    def __call__(self, text):
        ids = self.encode(text)[:16]
        return torch.tensor([ids + [0] * (16 - len(ids))])


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(64, 4)
        with torch.no_grad():
            values = torch.arange(256, dtype=torch.float32).view(64, 4)
            self.embeddings.word_embeddings.weight.copy_(torch.sin(values * 0.7) + 0.4)
            self.embeddings.word_embeddings.weight[0].zero_()

    @property
    def dtype(self):
        return self.embeddings.word_embeddings.weight.dtype


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.visual.weight.copy_(torch.eye(4))
        self.text = nn.Module()
        self.text.transformer = TinyTransformer()
        self.logit_scale = nn.Parameter(torch.tensor(0.7))
        self._errorcompass_tokenizer = TinyTokenizer()
        self.requires_grad_(False)

    def encode_text(self, text, inputs_embeds=False, y=None, normalize=False):
        embeddings = text if inputs_embeds else self.text.transformer.embeddings.word_embeddings(text)
        tokens = y if inputs_embeds else text
        mask = tokens.ne(0).unsqueeze(-1)
        features = (embeddings * mask).sum(1) / mask.sum(1)
        return F.normalize(features, dim=-1) if normalize else features


NAMES = ["alpha", "beta", "gamma"]
TEMPLATES = {name: [f"a scan of {name}.", f"an image shows {name}.", f"a tissue sample of {name}."]
             for name in NAMES}
IMAGES = torch.tensor([[1.0, 0.2, 0.1, 0.0], [0.1, 1.0, 0.2, 0.0],
                       [0.0, 0.1, 1.0, 0.3], [0.3, 0.2, 0.1, 1.0]])
LABELS = torch.tensor([0, 1, 2, 0])


def make_model(classnames=None, support=None, **options):
    torch.manual_seed(31)
    cfg = configuration(**options)
    backbone = TinyBackbone()
    classnames = NAMES if classnames is None else classnames
    if support is None:
        support = [{"img": IMAGES.clone(), "label": LABELS.clone()}]
    templates = {name: [f"a scan of {name}.", f"an image shows {name}.", f"a tissue sample of {name}."]
                 for name in classnames}
    with patch.object(implementation, "BIOMEDCOOP_TEMPLATES", templates), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = implementation.ErrorCompassModel(cfg, classnames, backbone, support)
    return model.train()


class ModelIntegrationTests(unittest.TestCase):
    def test_missing_support_class_fails_instead_of_silently_substituting_text(self):
        keep = LABELS != 2
        with self.assertRaisesRegex(ValueError, 'requires labeled support'):
            make_model(support=[{"img": IMAGES[keep], "label": LABELS[keep]}])

    def test_multitoken_class_length_and_middle_prompt_construction(self):
        model = make_model(classnames=["alpha tissue", "beta", "gamma"], CLASS_TOKEN_POSITION="middle")
        self.assertEqual(model.prompt_learner.name_lens, [2, 1, 1])
        self.assertTrue(torch.isfinite(model(IMAGES, LABELS)[0]).all())

    def test_real_trainer_forward_backward_handles_fp32_and_cpu_amp(self):
        for precision in ("fp32", "amp"):
            with self.subTest(precision=precision):
                trainer = implementation.ErrorCompass_BiomedCLIP()
                trainer.model = make_model(PREC=precision)
                trainer.cfg = trainer.model.cfg
                trainer.device = torch.device("cpu")
                trainer.optim = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
                trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
                trainer.epoch = 0
                trainer.batch_idx, trainer.num_batches = 0, 2
                before = trainer.model.prompt_learner.ctx.detach().clone()
                with patch.object(implementation, "compute_accuracy", _accuracy_module.compute_accuracy):
                    summary = trainer.forward_backward({"img": IMAGES, "label": LABELS})
                self.assertEqual(set(summary), {"loss", "acc", "loss_ce", "loss_sccm",
                                                "loss_kl", "loss_chc", "loss_cpmc"})
                self.assertGreater((trainer.model.prompt_learner.ctx - before).abs().sum().item(), 0)

    def test_default_forward_and_backward_use_actual_model(self):
        model = make_model()
        output = model(IMAGES, LABELS)
        self.assertEqual(len(output), 6)
        self.assertEqual(output[0].shape, (4, 3))
        for loss in output[1:]:
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertTrue(torch.isfinite(loss))
        sum(output[1:]).backward()
        for parameter in (model.prompt_learner.ctx,
                          model.prompt_learner.scgp_propagation.propagation_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.image_encoder.parameters()))
        self.assertFalse(model.image_encoder.training)
        self.assertFalse(model.text_encoder.training)

    def test_sccm_uses_prompt_features_before_fusion(self):
        model = make_model()
        with torch.no_grad():
            model.prompt_learner.scale_mm.fill_(1.0)
            prompt = F.normalize(model.text_encoder(model.prompt_learner(), model.tokenized_prompts), dim=-1)
            fused, _ = model.prompt_learner.apply_scgp(
                prompt, model.base_text_features, model.base_img_features)
            expected = ((prompt - model.base_text_features) ** 2).sum(1).mean() * 0.7
            incorrect_fused = ((fused - model.base_text_features) ** 2).sum(1).mean() * 0.7
        self.assertGreater(abs(expected.item() - incorrect_fused.item()), 1e-4)
        actual = model(IMAGES, LABELS)[2]
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(model.prompt_learner.ctx.grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.prompt_learner.scgp_propagation.propagation_weight.grad)

    def test_full_model_updates_shared_hard_count_feedback(self):
        model = make_model(CONF_UPDATE_FREQ=1)
        output = model(IMAGES, LABELS)
        self.assertGreater(output[4].item(), 0.0)
        predictions = output[0].argmax(1)
        expected = torch.zeros(3, 3)
        for label, prediction in zip(LABELS, predictions):
            expected[label, prediction] += 0.1
        torch.testing.assert_close(model.prompt_learner.confusion_matrix, expected)
        self.assertEqual(model.prompt_learner.chc_step_counter.item(), 1)
        self.assertGreater(model.prompt_learner.confusion_matrix.sum().item(), 0.0)

    def test_eval_does_not_update_feedback_or_require_labels(self):
        model = make_model(CONF_UPDATE_FREQ=1)
        model(IMAGES, LABELS)
        matrix = model.prompt_learner.confusion_matrix.clone()
        adjacency = model.prompt_learner.confusion_adjacency.clone()
        step = model.prompt_learner.chc_step_counter.clone()
        model.eval()
        with torch.no_grad():
            logits = model(IMAGES)
        self.assertEqual(logits.shape, (4, 3))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(model.prompt_learner.confusion_matrix, matrix)
        torch.testing.assert_close(model.prompt_learner.confusion_adjacency, adjacency)
        torch.testing.assert_close(model.prompt_learner.chc_step_counter, step)

    def test_cpmc_respects_shared_warmup_and_empty_pair_filter(self):
        for epoch, confusion in [(4, 0.8), (20, 0.1)]:
            model = make_model()
            model.current_epoch = epoch
            with torch.no_grad():
                model.prompt_learner.confusion_adjacency.fill_(confusion)
                model.prompt_learner.confusion_adjacency.fill_diagonal_(0)
            loss = model(IMAGES, LABELS)[5]
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertEqual(model.prompt_learner.scgp_propagation.propagation_weight.grad.abs().sum().item(), 0.0)

    def test_cpmc_uses_propagated_prototypes(self):
        model = make_model()
        model.current_epoch = 20
        learner = model.prompt_learner
        with torch.no_grad():
            learner.confusion_adjacency.fill_(0.8)
            learner.confusion_adjacency.fill_diagonal_(0)
            prompt = F.normalize(model.text_encoder(learner(), model.tokenized_prompts), dim=-1)
            _, propagated = learner.apply_scgp(
                prompt, model.base_text_features, model.base_img_features,
                current_epoch=model.current_epoch)
            args = (learner.confusion_adjacency, learner.cpmc_top_k,
                    learner.cpmc_threshold, learner.cpmc_margin_base, learner.cpmc_margin_scale)
            expected = implementation.cpmc_loss(propagated, *args) * learner.lambda_cpmc
            prompt_loss = implementation.cpmc_loss(prompt, *args) * learner.lambda_cpmc
        self.assertGreater(abs(expected.item() - prompt_loss.item()), 1e-4)
        actual = model(IMAGES, LABELS)[5]
        torch.testing.assert_close(actual, expected)
        actual.backward()
        gradient = learner.scgp_propagation.propagation_weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertIsNone(learner.ctx.grad)

    def test_multilayer_scgp_preserves_class_output_and_gradients(self):
        model = make_model(SCGP_NUM_LAYERS=2)
        layers = model.prompt_learner.scgp_propagation_layers
        with torch.no_grad():
            for layer in layers:
                layer.propagation_weight.copy_(torch.eye(4) + 0.1)
        output = model(IMAGES, LABELS)
        self.assertEqual(output[0].shape, (4, 3))
        sum(output[1:]).backward()
        self.assertEqual(len(layers), 2)
        for layer in layers:
            gradient = layer.propagation_weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_training_requires_labels(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, "requires labels"):
            model(IMAGES)

    def test_current_version_is_required_before_checkpoint_loading(self):
        validate = implementation.ErrorCompass_BiomedCLIP._validate_algorithm_version
        current = make_model().state_dict()
        self.assertEqual(int(current["algorithm_version"]), 3)
        validate(current)
        old = {key: value for key, value in current.items() if key != "algorithm_version"}
        with self.assertRaisesRegex(RuntimeError, "predates"):
            validate(old)
        for unsupported in (1, 2, 4):
            with self.subTest(version=unsupported), self.assertRaises(RuntimeError):
                validate(dict(current, algorithm_version=torch.tensor(unsupported)))


if __name__ == "__main__":
    unittest.main()
