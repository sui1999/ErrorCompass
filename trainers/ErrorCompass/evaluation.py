"""Cross-dataset evaluation with labeled target-training prototypes.

The dataset builders are imported lazily so the evaluation protocol can be
tested without downloading BiomedCLIP or importing the training stack.
"""

import torch


class CrossDatasetEvaluationMixin:
    """Evaluate independent target models without changing the source trainer.

    ``build_target_model(target_cfg, classnames, support_loader)`` constructs
    a fresh model for each target. All labeled target-training images define
    fixed visual prototypes; source-trained parameters are unchanged. No target
    gradient optimization or confusion updates are performed. Target test images
    are used only for evaluation. Source-shot fusion coefficients are retained.
    """

    @staticmethod
    def _cross_dataset_builders():
        from dassl.data.datasets import DATASET_REGISTRY, build_dataset
        from dassl.data.data_manager import build_data_loader
        from dassl.data.transforms import build_transform

        return DATASET_REGISTRY, build_dataset, build_data_loader, build_transform

    @staticmethod
    def _assert_independent_target(source_model, target_model):
        if not isinstance(target_model, torch.nn.Module):
            raise TypeError("build_target_model must return a torch.nn.Module")
        source_modules = {id(module) for module in source_model.modules()}
        if any(id(module) in source_modules for module in target_model.modules()):
            raise ValueError("Target model must not share modules with the source model")

        # A shallow parameter copy can share storage despite distinct modules.
        source_storage = {
            (tensor.device, tensor.untyped_storage().data_ptr())
            for tensor in list(source_model.parameters()) + list(source_model.buffers())
            if tensor.numel()
        }
        for tensor in list(target_model.parameters()) + list(target_model.buffers()):
            if tensor.numel() and (tensor.device, tensor.untyped_storage().data_ptr()) in source_storage:
                raise ValueError("Target model must not share tensor storage with the source model")

    @torch.no_grad()
    def cross_dataset_evaluate(self, target_datasets):
        """Return target test accuracies (%) using source-trained weights only.

        Names must be exact registered dataset names. Duplicate names are
        removed in input order and the source dataset is skipped. Failures are
        propagated instead of reporting an incomplete result as successful.
        The source model, config, evaluator and optimizer are never reassigned
        or used for inference/training by this method.
        """
        if isinstance(target_datasets, (str, bytes)):
            raise TypeError("target_datasets must be an iterable of registered dataset names")
        requested = list(target_datasets)
        if not requested:
            raise ValueError("At least one target dataset is required")
        if any(not isinstance(name, str) or not name for name in requested):
            raise ValueError("Each target dataset must be a nonempty registered name")

        registry, build_dataset, build_data_loader, build_transform = self._cross_dataset_builders()
        available = set(registry.registered_names())
        invalid = sorted(set(requested) - available)
        if invalid:
            raise ValueError(
                "Unknown target dataset(s): {}. Registered names: {}".format(
                    ", ".join(invalid), ", ".join(sorted(available))
                )
            )
        targets = list(dict.fromkeys(name for name in requested if name != self.cfg.DATASET.NAME))
        if not targets:
            raise ValueError("No target datasets remain after skipping the source dataset")

        results = {}
        for name in targets:
            target_cfg = self.cfg.clone()
            target_cfg.defrost()
            target_cfg.DATASET.NAME = name
            target_cfg.DATASET.SUBSAMPLE_CLASSES = "all"
            target_cfg.freeze()

            dataset_cfg = target_cfg.clone()
            dataset_cfg.defrost()
            dataset_cfg.DATASET.NUM_SHOTS = -1
            dataset_cfg.freeze()
            dataset = build_dataset(dataset_cfg)
            classnames = list(dataset.classnames)
            if not classnames:
                raise ValueError("Target dataset {} has no class names".format(name))
            transform = build_transform(dataset_cfg, is_train=False)
            loader = build_data_loader(
                dataset_cfg,
                sampler_type="SequentialSampler",
                data_source=dataset.test,
                batch_size=dataset_cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=transform,
                is_train=False,
            )

            support_loader = build_data_loader(
                dataset_cfg,
                sampler_type="SequentialSampler",
                data_source=dataset.train_x,
                batch_size=dataset_cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=transform,
                is_train=False,
            )
            target_model = self.build_target_model(target_cfg, classnames, support_loader)
            self._assert_independent_target(self.model, target_model)
            target_model.to(self.device)
            target_model.eval()
            correct, total = 0, 0
            for batch in loader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = target_model(images)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError("Target model must return a [batch, classes] logits tensor")
                if logits.shape != (labels.numel(), len(classnames)) or labels.ndim != 1:
                    raise ValueError("Target logits do not match the test labels and class names")
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Non-finite target logits for {}".format(name))
                if ((labels < 0) | (labels >= len(classnames))).any().item():
                    raise ValueError("Target labels are outside the class-name range")
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.numel()

            if total == 0:
                raise ValueError("Target dataset {} has no test examples".format(name))
            results[name] = 100.0 * correct / total
            print("[Cross-Dataset] {}: {:.2f}% ({}/{})".format(name, results[name], correct, total))
            # Release each target model before constructing the next one.
            del target_model

        return results
