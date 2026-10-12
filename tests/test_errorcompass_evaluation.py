"""Protocol tests with real torch and fake dataset/model construction only."""

import copy
import ast
import importlib.util
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


SPEC = importlib.util.spec_from_file_location(
    "errorcompass_evaluation",
    Path(__file__).resolve().parents[1] / "trainers" / "ErrorCompass" / "evaluation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CrossDatasetEvaluationMixin = MODULE.CrossDatasetEvaluationMixin


class Config(SimpleNamespace):
    def clone(self):
        return copy.deepcopy(self)

    def defrost(self):
        self.frozen = False

    def freeze(self):
        self.frozen = True


class TargetDataset:
    classnames = ("normal", "abnormal")

    def __init__(self):
        self._train_x = [{"img": torch.tensor([[3.0, 0.0], [0.0, 3.0]]), "label": torch.tensor([0, 1])}]
        self.test = [
            {"img": torch.tensor([[2.0, 0.0], [0.0, 2.0]]), "label": torch.tensor([0, 1])},
            {"img": torch.tensor([[2.0, 0.0]]), "label": torch.tensor([1])},
        ]

    @property
    def train_x(self):
        return self._train_x

    @property
    def val(self):
        raise AssertionError("Target validation examples must not be read")


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))
        self.dropout = nn.Dropout()
        self.register_buffer("feedback", torch.tensor([3.0]))
        self.calls = []

    def forward(self, images):
        self.calls.append((self.training, torch.is_grad_enabled()))
        if self.training:
            raise AssertionError("Target inference must run in eval mode")
        return images @ self.weight


class Harness(CrossDatasetEvaluationMixin):
    def __init__(self):
        self.cfg = Config(
            DATASET=SimpleNamespace(NAME="Source", NUM_SHOTS=16, SUBSAMPLE_CLASSES="base"),
            DATALOADER=SimpleNamespace(TEST=SimpleNamespace(BATCH_SIZE=2)),
            frozen=True,
        )
        self.device = torch.device("cpu")
        self.model = ToyModel()
        self.model.dropout.eval()  # Mixed source training flags must be preserved.
        self.evaluator = object()
        self.optim = object()
        self.datasets = []
        self.loaders = []
        self.models = []
        self.factory_configs = []
        self.factory_source_weights = []
        self.fail_target = None

    def _cross_dataset_builders(self):
        registry = SimpleNamespace(registered_names=lambda: ["Source", "TargetA", "TargetB"])

        def build_dataset(cfg):
            if cfg.DATASET.NAME == self.fail_target:
                raise RuntimeError("dataset unavailable")
            dataset = TargetDataset()
            self.datasets.append((cfg, dataset))
            return dataset

        def build_transform(cfg, is_train):
            if is_train:
                raise AssertionError("Target training transform must not be constructed")
            return "test_transform"

        def build_data_loader(cfg, **kwargs):
            self.loaders.append(kwargs)
            dataset = self.datasets[-1][1]
            if kwargs["data_source"] is not dataset.test and kwargs["data_source"] is not dataset.train_x:
                raise AssertionError("Only target train prototypes and test evaluation are permitted")
            if kwargs["is_train"]:
                raise AssertionError("Target train loader must not be constructed")
            return kwargs["data_source"]

        return registry, build_dataset, build_data_loader, build_transform

    def build_target_model(self, cfg, classnames, support_loader):
        if classnames != ["normal", "abnormal"]:
            raise AssertionError("Target class names must be passed to the factory")
        if support_loader is not self.datasets[-1][1].train_x:
            raise AssertionError('Factory must receive the full labeled target-training split')
        self.factory_configs.append(cfg)
        self.factory_source_weights.append(self.model.weight.detach().clone())
        model = copy.deepcopy(self.model)
        model.feedback.zero_()
        self.models.append(model)
        return model


class CrossDatasetEvaluationTests(unittest.TestCase):
    def test_target_train_prototypes_test_evaluation_and_independent_source_weights(self):
        trainer = Harness()
        source_state = {key: value.clone() for key, value in trainer.model.state_dict().items()}
        source_modes = [module.training for module in trainer.model.modules()]
        cfg_before = copy.deepcopy(vars(trainer.cfg))
        source_model, evaluator, optimizer = trainer.model, trainer.evaluator, trainer.optim

        with torch.enable_grad():
            results = trainer.cross_dataset_evaluate(["Source", "TargetA", "TargetA", "TargetB"])
            self.assertTrue(torch.is_grad_enabled())

        self.assertEqual(list(results), ["TargetA", "TargetB"])
        self.assertAlmostEqual(results["TargetA"], 200.0 / 3.0)
        self.assertAlmostEqual(results["TargetB"], 200.0 / 3.0)
        self.assertIs(trainer.model, source_model)
        self.assertIs(trainer.evaluator, evaluator)
        self.assertIs(trainer.optim, optimizer)
        self.assertEqual(vars(trainer.cfg), cfg_before)
        self.assertEqual([module.training for module in trainer.model.modules()], source_modes)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, source_state[key])
        self.assertEqual(trainer.model.calls, [])
        self.assertEqual(len(trainer.models), 2)
        self.assertIsNot(trainer.models[0], trainer.models[1])
        for index, model in enumerate(trainer.models):
            self.assertEqual(model.calls, [(False, False), (False, False)])
            self.assertFalse(model.training)
            torch.testing.assert_close(trainer.factory_source_weights[index], source_state["weight"])
            self.assertEqual(model.feedback.item(), 0)
            self.assertEqual(trainer.factory_configs[index].DATASET.NUM_SHOTS, 16)
            self.assertEqual(trainer.factory_configs[index].DATASET.SUBSAMPLE_CLASSES, "all")
            self.assertTrue(trainer.factory_configs[index].frozen)
            self.assertEqual(trainer.datasets[index][0].DATASET.NUM_SHOTS, -1)
            self.assertEqual(trainer.loaders[2 * index]["sampler_type"], "SequentialSampler")
            self.assertEqual(trainer.loaders[2 * index]["tfm"], "test_transform")
        trainer.models[0].weight.data.fill_(8)
        torch.testing.assert_close(trainer.models[1].weight, source_state["weight"])
        torch.testing.assert_close(trainer.model.weight, source_state["weight"])

    def test_no_targets_and_only_source_are_rejected(self):
        for targets in ([], ["Source"], ["Source", "Source"]):
            with self.subTest(targets=targets):
                trainer = Harness()
                with self.assertRaises(ValueError):
                    trainer.cross_dataset_evaluate(targets)
                self.assertEqual(trainer.datasets, [])
                self.assertEqual(trainer.models, [])

    def test_invalid_names_fail_before_any_dataset_is_built(self):
        for targets in (["TargetA", "unknown"], [""], [None]):
            trainer = Harness()
            with self.assertRaises(ValueError):
                trainer.cross_dataset_evaluate(targets)
            self.assertEqual(trainer.datasets, [])
        with self.assertRaises(TypeError):
            Harness().cross_dataset_evaluate("TargetA")

    def test_dataset_failure_propagates(self):
        trainer = Harness()
        trainer.fail_target = "TargetB"
        with self.assertRaisesRegex(RuntimeError, "dataset unavailable"):
            trainer.cross_dataset_evaluate(["TargetA", "TargetB"])
        self.assertEqual(trainer.cfg.DATASET.NAME, "Source")
        self.assertTrue(trainer.model.training)
        self.assertEqual(trainer.model.feedback.item(), 3)

    def test_shared_source_model_or_parameter_storage_is_rejected(self):
        for alias_storage in (False, True):
            trainer = Harness()

            def unsafe_factory(cfg, classnames, support_loader):
                if alias_storage:
                    model = ToyModel()
                    model.weight = nn.Parameter(trainer.model.weight.detach())
                    return model
                return trainer.model

            trainer.build_target_model = unsafe_factory
            with self.assertRaisesRegex(ValueError, "must not share"):
                trainer.cross_dataset_evaluate(["TargetA"])
            self.assertTrue(trainer.model.training)
            self.assertEqual(trainer.model.calls, [])


class FactoryBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 3, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self._errorcompass_tokenizer = {"source_asset": "offline_tokenizer"}


class FactoryModel(nn.Module):
    """Small stand-in for expensive model construction, not factory logic."""

    def __init__(self, cfg, classnames, backbone, train_loader_x=None, text_only=False, transfer=False):
        super().__init__()
        if transfer and (train_loader_x is None or text_only):
            raise AssertionError("Cross-domain factory must use target-training prototypes")
        self.support_loader = train_loader_x
        self.cfg = cfg
        self.dtype = torch.float32
        self.current_epoch = 0
        self.text_encoder = nn.Module()
        self.text_encoder.model = backbone
        self.image_encoder = backbone.visual
        self.logit_scale = backbone.logit_scale
        self.register_buffer("algorithm_version", torch.tensor(3))
        self.prompt_learner = nn.Module()
        self.prompt_learner.ctx = nn.Parameter(torch.full((2, 3), -1.0))
        self.prompt_learner.scgp_propagation_layers = nn.ModuleList([
            nn.Linear(3, 3, bias=False) for _ in range(cfg.TRAINER.ERRORCOMPASS.SCGP_NUM_LAYERS)
        ])
        self.prompt_learner.scale_mm = nn.Parameter(torch.ones(1))
        self.prompt_learner.register_buffer('alpha_it', torch.tensor(0.7))
        self.prompt_learner.register_buffer('beta_it', torch.tensor(0.3))
        self.prompt_learner.scgp_alpha_scale = nn.Parameter(torch.tensor(0.3))
        self.prompt_learner.scgp_beta_scale = nn.Parameter(torch.tensor(0.2))
        self.prompt_learner.ZS_image_encoder = backbone.visual
        nclasses = len(classnames)
        anchors = backbone.visual.weight.detach().mean(dim=0).repeat(nclasses, 1)
        self.register_buffer("base_text_features", anchors.clone())
        self.register_buffer("base_img_features", anchors.clone())
        self.prompt_learner.register_buffer("fixed_embeddings", anchors[:, None].clone(), persistent=False)
        for name in ("token_prefix", "token_suffix"):
            self.prompt_learner.register_buffer(name, torch.zeros(nclasses, 1, 3))
        for name in ("confusion_matrix", "confusion_adjacency"):
            self.prompt_learner.register_buffer(name, torch.zeros(nclasses, nclasses))
        self.prompt_learner.register_buffer("chc_step_counter", torch.tensor(0))
        for name in ("class_reliability", "class_hardness", "hardness_count"):
            self.prompt_learner.register_buffer(name, torch.zeros(nclasses))


def load_real_trainer_methods():
    """Execute the actual trainer class with expensive dependencies stubbed.

    Only class bases/decorators are removed. The factory, checkpoint-validation
    and load methods under test remain unchanged from the implementation.
    """
    path = Path(__file__).resolve().parents[1] / "trainers" / "ErrorCompass" / "errorcompass_biomedclip.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ErrorCompass_BiomedCLIP")
    version = next(node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "ALGORITHM_VERSION" for target in node.targets))
    trainer.bases = []
    trainer.decorator_list = []
    namespace = {
        "torch": torch, "nn": nn, "copy": copy, "osp": osp,
        "ErrorCompassModel": FactoryModel,
    }
    module = ast.Module(body=[version, trainer], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["ErrorCompass_BiomedCLIP"], namespace


class TargetFactoryAndCheckpointTests(unittest.TestCase):
    def make_trainer(self, nclasses=2):
        trainer_class, namespace = load_real_trainer_methods()
        trainer = trainer_class()
        trainer.device = torch.device("cpu")
        trainer.cfg = Config(
            DATASET=SimpleNamespace(NAME="Source", NUM_SHOTS=16, SUBSAMPLE_CLASSES="all"),
            TRAINER=SimpleNamespace(ERRORCOMPASS=SimpleNamespace(CHECKPOINT_DIR="unused", SCGP_NUM_LAYERS=3)),
            OPTIM=SimpleNamespace(MAX_EPOCH=100), frozen=True,
        )
        trainer.model = FactoryModel(trainer.cfg, [str(i) for i in range(nclasses)], FactoryBackbone(), text_only=True)
        trainer._models = {"prompt_learner": trainer.model}
        trainer.get_model_names = lambda: ["prompt_learner"]
        with torch.no_grad():
            trainer.model.image_encoder.weight.fill_(7)
            trainer.model.logit_scale.fill_(8)
            trainer.model.prompt_learner.ctx.fill_(9)
            for index, layer in enumerate(trainer.model.prompt_learner.scgp_propagation_layers):
                layer.weight.fill_(10 + index)
            trainer.model.prompt_learner.scale_mm.fill_(4)
            trainer.model.prompt_learner.confusion_matrix.fill_(5)
            trainer.model.prompt_learner.confusion_adjacency.fill_(6)
            trainer.model.prompt_learner.chc_step_counter.fill_(99)
        trainer.model.current_epoch = 100
        return trainer, namespace

    def test_actual_factory_transfers_all_graph_layers_and_source_backbone(self):
        trainer, namespace = self.make_trainer()

        def forbidden_reload(*args, **kwargs):
            raise AssertionError("Target must use the loaded source backbone, not reload external weights")

        namespace["load_biomedclip_from_local"] = forbidden_reload
        before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
        target_cfg = trainer.cfg.clone()
        target_cfg.DATASET.NAME = "TargetA"
        support = object()
        target = trainer.build_target_model(target_cfg, ["class_a", "class_b", "class_c"], support)
        self.assertIs(target.support_loader, support)
        CrossDatasetEvaluationMixin._assert_independent_target(trainer.model, target)
        self.assertEqual(target.cfg.DATASET.NUM_SHOTS, 16)
        self.assertEqual(target.current_epoch, 0)
        self.assertFalse(target.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in target.parameters()))
        torch.testing.assert_close(target.image_encoder.weight, trainer.model.image_encoder.weight)
        torch.testing.assert_close(target.logit_scale, trainer.model.logit_scale)
        torch.testing.assert_close(target.prompt_learner.ctx, trainer.model.prompt_learner.ctx)
        for source_layer, target_layer in zip(trainer.model.prompt_learner.scgp_propagation_layers,
                                             target.prompt_learner.scgp_propagation_layers):
            torch.testing.assert_close(target_layer.weight, source_layer.weight)
        torch.testing.assert_close(target.base_text_features, target.base_img_features)
        self.assertEqual(target.base_text_features.shape, (3, 3))
        self.assertTrue((target.base_text_features == 7).all().item())
        self.assertEqual(target.prompt_learner.confusion_matrix.sum().item(), 0)
        self.assertEqual(target.prompt_learner.confusion_adjacency.sum().item(), 0)
        self.assertEqual(target.prompt_learner.chc_step_counter.item(), 0)
        self.assertIsNot(target.text_encoder.model._errorcompass_tokenizer,
                         trainer.model.text_encoder.model._errorcompass_tokenizer)
        self.assertTrue(trainer.model.training)
        self.assertEqual(trainer.model.current_epoch, 100)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_actual_factory_rejects_class_specific_source_prompts(self):
        trainer, _ = self.make_trainer()
        trainer.model.prompt_learner.ctx = nn.Parameter(torch.zeros(2, 2, 3))
        with self.assertRaisesRegex(ValueError, "CSC=False"):
            trainer.build_target_model(trainer.cfg.clone(), ["a", "b"], object())

    def test_checkpoint_version_and_same_domain_epoch_are_enforced(self):
        trainer, namespace = self.make_trainer()
        source_state = copy.deepcopy(trainer.model.state_dict())
        paths = []

        def load_checkpoint(path):
            paths.append(path)
            return {"state_dict": source_state, "epoch": 100}

        namespace["load_checkpoint"] = load_checkpoint
        trainer.model.current_epoch = 0
        trainer.load_model("source_output")
        # Dassl saves one-based completed epochs, while graph warmup consumes
        # the zero-based epoch used by forward_backward during training.
        self.assertEqual(trainer.model.current_epoch, 99)
        self.assertEqual(paths, [osp.join("source_output", "prompt_learner", "model.pth.tar-100")])
        namespace["load_checkpoint"] = lambda path: {"state_dict": source_state, "epoch": 5}
        trainer.load_model("source_output", epoch=5)
        self.assertEqual(trainer.model.current_epoch, 4)
        for version in (None, 1, 2, 4):
            with self.subTest(version=version):
                legacy = copy.deepcopy(source_state)
                if version is None:
                    del legacy["algorithm_version"]
                else:
                    legacy["algorithm_version"] = torch.tensor(version)
                namespace["load_checkpoint"] = lambda path: {"state_dict": legacy, "epoch": 100}
                with self.assertRaisesRegex(RuntimeError, "paper-aligned"):
                    trainer.load_model("source_output")

    def test_novel_checkpoint_load_keeps_target_class_state_and_resets_epoch(self):
        source, _ = self.make_trainer(nclasses=2)
        target, namespace = self.make_trainer(nclasses=3)
        target.cfg.DATASET.SUBSAMPLE_CLASSES = "new"
        target.model.base_text_features.fill_(21)
        target.model.base_img_features.fill_(21)
        target.model.prompt_learner.confusion_matrix.zero_()
        target.model.prompt_learner.confusion_adjacency.zero_()
        target.model.prompt_learner.chc_step_counter.zero_()
        state = copy.deepcopy(source.model.state_dict())
        namespace["load_checkpoint"] = lambda path: {"state_dict": state, "epoch": 100}
        target.load_model("source_output", epoch=100)
        self.assertEqual(target.model.current_epoch, 0)
        self.assertTrue((target.model.base_img_features == 21).all().item())
        self.assertEqual(target.model.prompt_learner.confusion_matrix.sum().item(), 0)
        self.assertEqual(target.model.prompt_learner.chc_step_counter.item(), 0)
        torch.testing.assert_close(target.model.prompt_learner.ctx, source.model.prompt_learner.ctx)
        for source_layer, target_layer in zip(source.model.prompt_learner.scgp_propagation_layers,
                                             target.model.prompt_learner.scgp_propagation_layers):
            torch.testing.assert_close(target_layer.weight, source_layer.weight)


if __name__ == "__main__":
    unittest.main()
