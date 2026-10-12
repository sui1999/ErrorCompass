import copy
from tqdm import tqdm
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES
from .graph import GraphConvolution, semantic_adjacency, context_adjacency
from .losses import (confusion_adjacency, update_confusion_counts,
                     sccm_loss, cpmc_loss, kl_teacher_loss, chc_loss)
from .evaluation import CrossDatasetEvaluationMixin

ALGORITHM_VERSION = 3

import json
import os
from open_clip.src.open_clip.factory import create_model_and_transforms, HF_HUB_PREFIX, _MODEL_CONFIGS
from open_clip.src.open_clip.tokenizer import HFTokenizer


def load_biomedclip_from_local(checkpoint_dir="checkpoints"):
    model_name = "biomedclip_local"
    config_path = os.path.join(checkpoint_dir, "open_clip_config.json")
    model_path = os.path.join(checkpoint_dir, "open_clip_pytorch_model.bin")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Local checkpoint files not found at {checkpoint_dir}. "
                                f"Expected open_clip_config.json and open_clip_pytorch_model.bin")

    with open(config_path, "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]


    if "text_cfg" in model_cfg and "hf_model_name" in model_cfg["text_cfg"]:
        hf_name = model_cfg["text_cfg"]["hf_model_name"]
        print(f"Detected HF underlying text model: {hf_name}")


        local_bert_path = os.path.join(checkpoint_dir, "bert")


        if os.path.exists(os.path.join(local_bert_path, "config.json")):
            print(f" Redirecting text tower to local path: {local_bert_path}")
            model_cfg["text_cfg"]["hf_model_name"] = local_bert_path


            if "tokenizer_name" in model_cfg["text_cfg"]:
                model_cfg["text_cfg"]["tokenizer_name"] = local_bert_path
        else:
            print(f" Warning: Local BERT config not found at {local_bert_path}. "
                  f"Expect setup connection error if offline.")


    _MODEL_CONFIGS[model_name] = model_cfg

    print(f"Loading tokenizer from local directory: {checkpoint_dir}...")


    try:


        context_length = model_cfg.get("text_cfg", {}).get("context_length", 256)
        tokenizer = HFTokenizer(checkpoint_dir, context_length=context_length)
    except Exception as e:
        print(f"Failed to load tokenizer from {checkpoint_dir}: {e}")
        print("Falling back to loading from original HF Hub name (requires internet)...")


        tokenizer_name = model_cfg.get("text_cfg", {}).get("tokenizer_name",
                                                           "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        from open_clip.src.open_clip.factory import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_name, context_length=context_length)

    print(f"Loading model and transforms from {model_path}...")
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=model_path,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    model._errorcompass_tokenizer = tokenizer
    print("Model loaded successfully.")
    return model, preprocess, tokenizer


def resolve_template_tau(proto_cfg, default=1.5):
    """Resolve the template outlier threshold."""
    return float(proto_cfg.get("TEMPLATE_TAU", default))


class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts, image_features=None):

        x = self.model.encode_text(prompts, True, tokenized_prompts)


        if image_features is not None and self.training:


            pass

        return x


def build_template_features(cfg, classnames, biomedclip_model):
    """Encode the fixed, offline template pool; no target image is required."""
    tokenizer = getattr(biomedclip_model, '_errorcompass_tokenizer', None)
    if tokenizer is None:
        raise ValueError("Backbone must be loaded with load_biomedclip_from_local().")
    names = [name.replace('_', ' ') for name in classnames]
    n_prompts = int(cfg.TRAINER.ERRORCOMPASS.N_PROMPTS)
    if n_prompts < 1:
        raise ValueError('N_PROMPTS must be positive.')
    for name in names:
        if name not in BIOMEDCOOP_TEMPLATES or len(BIOMEDCOOP_TEMPLATES[name]) < n_prompts:
            raise ValueError(f'Insufficient offline templates for class {name!r}.')
    device = next(biomedclip_model.parameters()).device
    with torch.no_grad():
        features = []
        for index in range(n_prompts):
            tokens = torch.cat([tokenizer(BIOMEDCOOP_TEMPLATES[name][index]) for name in names])
            features.append(biomedclip_model.encode_text(tokens.to(device)))
    return torch.stack(features, dim=1).detach()


def build_text_anchors(cfg, classnames, biomedclip_model, text_encoder=None, template_features=None):
    """T_c = normalize(mean_i f_txt(template_c,i)), as in the manuscript."""
    if template_features is None:
        template_features = build_template_features(cfg, classnames, biomedclip_model)
    return F.normalize(template_features.detach().mean(dim=1), dim=-1)


def build_visual_prototypes(cfg, classnames, biomedclip_model, img_encoder, train_loader_x):

    device = next(img_encoder.parameters()).device
    dtype = biomedclip_model.text.transformer.dtype
    img_encoder.eval()

    n_cls = len(classnames)

    with torch.no_grad():
        img_feature = []
        labels = []

        print('Extracting image features for graph construction...')
        for batch_idx, batch in tqdm(enumerate(train_loader_x),
                                     total=len(train_loader_x),
                                     desc='Building image graph'):
            image = batch["img"].to(device)
            label = batch["label"].to(device)

            image_features = img_encoder(image.type(dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            img_feature.append(image_features)
            labels.append(label)

        if not img_feature:
            raise ValueError('The support loader contains no images.')
        img_feature_list = torch.cat(img_feature, dim=0)
        label_list = torch.cat(labels, dim=0)


        sorted_labels, indices = torch.sort(label_list)
        img_feature_sorted = torch.index_select(img_feature_list, 0, indices)


        unique_labels, counts = torch.unique(sorted_labels, return_counts=True)
        num_classes_in_train = unique_labels.size(0)


        min_samples = counts.min().item()
        max_samples = counts.max().item()
        avg_samples = counts.float().mean().item()
        print(f"[Image Graph] Total classes: {n_cls}, Classes in train: {num_classes_in_train}")
        print(f"[Image Graph] Samples per class - Min: {min_samples}, Max: {max_samples}, Avg: {avg_samples:.1f}")

        if num_classes_in_train != n_cls or not torch.equal(
                unique_labels, torch.arange(n_cls, device=unique_labels.device)):
            raise ValueError('Every visual-prototype class requires labeled support images.')

        split_indices = counts.cumsum(dim=0)[:-1].tolist()
        features_split = torch.tensor_split(img_feature_sorted, split_indices, dim=0)


        class_means_dict = {}
        class_vars_dict = {}
        class_counts_dict = {}

        for idx, label_id in enumerate(unique_labels.tolist()):
            class_features = features_split[idx]
            class_means_dict[label_id] = class_features.mean(dim=0)
            class_counts_dict[label_id] = class_features.shape[0]

            if class_features.shape[0] > 1:
                class_vars_dict[label_id] = class_features.var(dim=0).mean().item()
            else:

                class_vars_dict[label_id] = 1.0


        img_feature_means_list = []

        feature_device = list(class_means_dict.values())[0].device if class_means_dict else device


        sample_counts = torch.zeros(n_cls, device=feature_device)
        class_variances = torch.ones(n_cls, device=feature_device)

        for class_id in range(n_cls):
            img_feature_means_list.append(class_means_dict[class_id])
            sample_counts[class_id] = class_counts_dict[class_id]
            class_variances[class_id] = class_vars_dict[class_id]

        img_feature_means = torch.stack(img_feature_means_list, dim=0)


        reliability_stats = {
            'sample_counts': sample_counts,
            'class_variances': class_variances,
            'min_samples': min_samples,
            'max_samples': max_samples,
            'avg_samples': avg_samples,
        }


        img_encoder = img_encoder.to(device)

    return img_feature_means.to(device), reliability_stats


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        opt = cfg.TRAINER.ERRORCOMPASS
        if opt.CSC:
            raise ValueError('The manuscript uses shared context vectors (CSC=False).')
        n_cls = len(classnames)
        if n_cls < 1:
            raise ValueError('At least one class is required.')
        n_ctx = opt.N_CTX
        dtype = biomedclip_model.text.transformer.dtype
        word_embeddings = biomedclip_model.text.transformer.embeddings.word_embeddings
        device = word_embeddings.weight.device
        self.tokenizer = biomedclip_model._errorcompass_tokenizer
        if cfg.INPUT.SIZE[0] != 224:
            raise ValueError('BiomedCLIP requires 224-pixel inputs.')
        ctx_init = opt.CTX_INIT
        if ctx_init and n_ctx == 4:
            prompt_prefix = ctx_init.replace('_', ' ')
            with torch.no_grad():
                embedding = word_embeddings(self.tokenizer(prompt_prefix).to(device)).to(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :].clone()
        else:
            shape = (n_cls, n_ctx, word_embeddings.embedding_dim) if opt.CSC else (n_ctx, word_embeddings.embedding_dim)
            ctx_vectors = torch.empty(shape, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = ' '.join(['X'] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)
        self.register_buffer('fixed_embeddings', build_template_features(cfg, classnames, biomedclip_model), persistent=False)
        self.ZS_image_encoder = biomedclip_model.visual
        self.hidden_dim = self.fixed_embeddings.shape[-1]
        self.use_scgp_adaptive_gate = opt.get('USE_SCGP_ADAPTIVE_GATE', False)
        self.feedback_enabled = True
        self.num_shots = cfg.DATASET.get('NUM_SHOTS', -1)
        coefficients = {1: (0.1, 0.1), 2: (0.3, 0.2), 4: (0.5, 0.3),
                        8: (0.7, 0.3), 16: (0.7, 0.3), -1: (0.7, 0.3)}
        if self.num_shots not in coefficients:
            raise ValueError('The manuscript defines fusion for 1, 2, 4, 8 and 16 shots.')
        alpha, beta = coefficients[self.num_shots]
        # Persistent buffers preserve source-shot coefficients in Novel loading.
        self.register_buffer('alpha_it', torch.tensor(alpha, device=device, dtype=dtype))
        self.register_buffer('beta_it', torch.tensor(beta, device=device, dtype=dtype))
        self.scgp_num_layers = int(opt.get('SCGP_NUM_LAYERS', 1))
        if self.scgp_num_layers < 1:
            raise ValueError('SCGP_NUM_LAYERS must be positive.')
        if self.scgp_num_layers == 1:
            self.scgp_propagation = GraphConvolution(self.hidden_dim).to(device=device, dtype=dtype)
        else:
            self.scgp_propagation_layers = nn.ModuleList([
                GraphConvolution(self.hidden_dim).to(device=device, dtype=dtype)
                for _ in range(self.scgp_num_layers)
            ])
        self.scale_mm = nn.Parameter(torch.ones(1, device=device, dtype=dtype))
        if self.use_scgp_adaptive_gate:
            self.scgp_reliability_temp = opt.get('SCGP_RELIABILITY_TEMP', 2.0)
            self.scgp_hardness_temp = opt.get('SCGP_HARDNESS_TEMP', 1.0)
            self.scgp_alpha_base = opt.get('SCGP_ALPHA_BASE', 0.5)
            self.scgp_beta_base = opt.get('SCGP_BETA_BASE', 0.3)
            self.scgp_gate_ema_rho = opt.get('SCGP_GATE_EMA_RHO', 0.9)
            self.register_buffer('class_reliability', torch.full((n_cls,), 0.5, device=device))
            self.register_buffer('class_hardness', torch.zeros(n_cls, device=device))
            self.register_buffer('hardness_count', torch.zeros(n_cls, device=device))
            self.scgp_alpha_scale = nn.Parameter(torch.tensor(0.3, device=device))
            self.scgp_beta_scale = nn.Parameter(torch.tensor(0.2, device=device))
        self.chc_top_k = opt.get('CHC_TOP_K', 3)
        self.chc_tau = float(opt.get('CHC_TAU', 1.0))
        if self.chc_tau <= 0:
            raise ValueError('CHC_TAU must be positive.')
        self.chc_beta = opt.get('CHC_BETA', 5.0)
        self.chc_eta = opt.get('CHC_ETA', 0.25)
        self.lambda_chc = opt.get('LAMBDA_CHC', 0.5)
        self.conf_ema_rho = opt.get('CONF_EMA_RHO', 0.9)
        self.scgp_semantic_weight = opt.get('SCGP_SEMANTIC_WEIGHT', 0.3)
        self.conf_warmup_epochs = opt.get('CONF_WARMUP_EPOCHS', 5)
        self.conf_update_freq = int(opt.get('CONF_UPDATE_FREQ', 10))
        if self.conf_update_freq < 1:
            raise ValueError('CONF_UPDATE_FREQ must be positive.')
        self.register_buffer('confusion_matrix', torch.zeros(n_cls, n_cls, device=device))
        self.register_buffer('confusion_adjacency', torch.zeros(n_cls, n_cls, device=device))
        self.register_buffer('chc_step_counter', torch.tensor(0, device=device))
        self.cpmc_margin_base = opt.get('CPMC_MARGIN_BASE', 0.5)
        self.cpmc_margin_scale = opt.get('CPMC_MARGIN_SCALE', 0.5)
        self.cpmc_threshold = opt.get('CPMC_THRESHOLD', 0.2)
        self.lambda_cpmc = opt.get('LAMBDA_CPMC', 0.3)
        self.cpmc_top_k = opt.get('CPMC_TOP_K', 5)
        names = [name.replace('_', ' ') for name in classnames]
        self.name_lens = [len(self.tokenizer.tokenizer.encode(name, add_special_tokens=False)) for name in names]
        prompts = [prompt_prefix + ' ' + name + '.' for name in names]
        self.tokenized_prompts = torch.cat([self.tokenizer(text) for text in prompts]).to(device)
        with torch.no_grad():
            embedding = word_embeddings(self.tokenized_prompts).to(dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = opt.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):


        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def apply_scgp(self, prompt_features, text_anchors, visual_prototypes, current_epoch=0):
        if self.use_scgp_adaptive_gate:
            alpha, beta = self.compute_scgp_adaptive_weights()
            alpha, beta = alpha[:, None], beta[:, None]
        else:
            alpha, beta = self.alpha_it, self.beta_it
        propagated = self.propagate_visual_prototypes(
            text_anchors, prompt_features, visual_prototypes, current_epoch=current_epoch
        )
        cross_modal = alpha * visual_prototypes + (1 - alpha) * propagated
        graph_branch = beta * text_anchors + (1 - beta) * cross_modal
        graph_weight = self.scale_mm.to(graph_branch.dtype).clamp(0.0, 1.5)
        fused = prompt_features + graph_weight * (graph_branch - prompt_features)
        return F.normalize(fused, dim=-1), propagated

    def compute_scgp_adaptive_weights(self):


        r = self.class_reliability
        h = self.class_hardness


        r_gate = torch.sigmoid((r - 0.5) / self.scgp_reliability_temp)


        h_gate = torch.sigmoid(h / self.scgp_hardness_temp)


        alpha_adjustment = self.scgp_alpha_scale * r_gate * (1 - h_gate)
        alpha_per_class = torch.clamp(self.scgp_alpha_base + alpha_adjustment, 0.05, 0.95)


        beta_adjustment = self.scgp_beta_scale * (1 - r_gate) * h_gate
        beta_per_class = torch.clamp(self.scgp_beta_base + beta_adjustment, 0.05, 0.95)

        return alpha_per_class, beta_per_class

    def init_reliability_from_stats(self, reliability_stats):

        if not self.use_scgp_adaptive_gate or reliability_stats is None:
            return

        sample_counts = reliability_stats['sample_counts']
        class_variances = reliability_stats['class_variances']


        device = self.class_reliability.device
        sample_counts = sample_counts.to(device)
        class_variances = class_variances.to(device)


        n_ref = max(self.num_shots, 4) if self.num_shots > 0 else 4
        count_reliability = torch.sigmoid((sample_counts / n_ref - 1) / 0.5)


        sigma_max = class_variances.max() + 1e-6
        variance_reliability = 1 - (class_variances / sigma_max)


        zero_sample_mask = (sample_counts < 0.5)
        combined_reliability = 0.6 * count_reliability + 0.4 * variance_reliability
        combined_reliability[zero_sample_mask] = 0.0

        self.class_reliability.copy_(combined_reliability)

        print(f"[SCGP Adaptive Gate] Initialized reliability scores: mean={combined_reliability.mean():.3f}, "
              f"min={combined_reliability.min():.3f}, max={combined_reliability.max():.3f}")

    def update_class_hardness(self, logits, labels):

        if not self.use_scgp_adaptive_gate:
            return

        with torch.no_grad():
            bs, n_cls = logits.shape


            s_pos = logits.gather(1, labels.unsqueeze(1)).squeeze(1)


            neg_mask = torch.ones_like(logits, dtype=torch.bool)
            neg_mask.scatter_(1, labels.unsqueeze(1), False)

            logits_neg = logits.clone()
            logits_neg[~neg_mask] = float('-inf')
            s_neg_max = logits_neg.max(dim=1)[0]


            margin = s_pos - s_neg_max


            hardness = F.relu(-margin)


            for i in range(bs):
                y = labels[i].item()
                if y < n_cls:

                    self.hardness_count[y] += 1
                    ema_weight = 1.0 / self.hardness_count[y].clamp(min=1, max=100)
                    self.class_hardness[y] = (1 - ema_weight) * self.class_hardness[y] + ema_weight * hardness[i]

    def propagate_visual_prototypes(self, text_anchors, prompt_features, visual_prototypes, current_epoch=0):
        with torch.no_grad():
            semantic = semantic_adjacency(text_anchors.detach())
            adjacency = self.build_context_adjacency(semantic, current_epoch=current_epoch)
        layers = self.scgp_propagation_layers if self.scgp_num_layers > 1 else [self.scgp_propagation]
        features = visual_prototypes
        for layer in layers:
            features = layer(features, adjacency.to(device=features.device, dtype=features.dtype))
        return features


    def update_confusion_matrix(self, logits, labels):
        if not self.training or not self.feedback_enabled:
            return
        with torch.no_grad():
            updated = update_confusion_counts(self.confusion_matrix, logits.detach(), labels, self.conf_ema_rho)
            self.confusion_matrix.copy_(updated)
            self.chc_step_counter.add_(1)
            if int(self.chc_step_counter.item()) % self.conf_update_freq == 0:
                self._update_confusion_adjacency()

    def _update_confusion_adjacency(self):
        with torch.no_grad():
            self.confusion_adjacency.copy_(confusion_adjacency(self.confusion_matrix))

    def build_context_adjacency(self, semantic_adjacency, current_epoch=0):
        active = self.feedback_enabled and current_epoch >= self.conf_warmup_epochs
        return context_adjacency(semantic_adjacency, self.confusion_adjacency,
                                 self.scgp_semantic_weight, feedback_active=active)


    def compute_chc_loss(self, logits, labels, current_epoch=0):
        active = self.feedback_enabled and current_epoch >= self.conf_warmup_epochs
        graph = self.confusion_adjacency if active else None
        return chc_loss(logits / self.chc_tau, labels, self.chc_top_k, self.chc_beta,
                        confusion=graph, eta=self.chc_eta)

    def compute_cpmc_loss(self, prototypes, current_epoch=0):
        if (not self.feedback_enabled
                or current_epoch < self.conf_warmup_epochs):
            return prototypes.sum() * 0.0
        return cpmc_loss(prototypes, self.confusion_adjacency, self.cpmc_top_k,
                         self.cpmc_threshold, self.cpmc_margin_base, self.cpmc_margin_scale)

    def forward(self, image_features=None, base_text_features=None, base_img_features=None):

        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class ErrorCompassModel(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model, train_loader_x=None, text_only=False, transfer=False):
        super().__init__()
        self.cfg = cfg
        biomedclip_model.requires_grad_(False).eval()
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.register_buffer('algorithm_version', torch.tensor(ALGORITHM_VERSION, dtype=torch.long))
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        anchors = build_text_anchors(cfg, classnames, biomedclip_model,
                                     template_features=self.prompt_learner.fixed_embeddings)
        self.register_buffer('base_text_features', anchors)
        self.text_only = text_only or str(cfg.DATASET.SUBSAMPLE_CLASSES).lower() == 'new'
        self.transfer = transfer
        if self.text_only or self.transfer:
            # Novel and transfer have no source-class confusion or target updates.
            self.prompt_learner.feedback_enabled = False
        if not self.text_only:
            if train_loader_x is None:
                raise ValueError('Visual prototypes require a labeled training/support loader.')
            visual, reliability = build_visual_prototypes(cfg, classnames, biomedclip_model,
                                                          self.image_encoder, train_loader_x)
            if self.prompt_learner.use_scgp_adaptive_gate:
                self.prompt_learner.init_reliability_from_stats(reliability)
        else:
            visual = anchors.detach().clone()
        self.register_buffer('base_img_features', visual.detach())
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.template_tau = resolve_template_tau(cfg.TRAINER.ERRORCOMPASS)
        self.current_epoch = 0

    def train(self, mode=True):
        super().train(mode)
        # Frozen encoders must also keep their dropout/batch-statistics frozen.
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.prompt_learner.ZS_image_encoder.eval()
        return self

    @torch.no_grad()
    def teacher_logits(self, image_features, logit_scale):
        # Preserve the baseline's template filtering; SCCM uses the full frozen anchor.
        templates = F.normalize(self.prompt_learner.fixed_embeddings, dim=-1)
        template_scores = logit_scale * torch.einsum('bd,cnd->bnc', image_features, templates)
        scores = template_scores.max(dim=-1).values.mean(dim=0)
        selected = templates.mean(dim=1)
        if scores.numel() > 1:
            median = scores.median()
            mad = (scores - median).abs().median()
            if torch.isfinite(mad) and mad > 1e-6:
                z = (scores - median) / mad
                std = z.std(unbiased=False)
                if torch.isfinite(std) and std > 1e-6:
                    keep = ((z - z.mean()) / std).abs() <= self.template_tau
                    if keep.any():
                        selected = templates[:, keep].mean(dim=1)
        return logit_scale * image_features @ F.normalize(selected, dim=-1).t()

    def forward(self, image, label=None):
        if self.training and (self.text_only or self.transfer):
            raise ValueError('Novel and cross-dataset target models are inference-only.')
        image_features = F.normalize(self.image_encoder(image.to(dtype=self.dtype)), dim=-1)
        prompts = self.prompt_learner()
        prompt_features = F.normalize(self.text_encoder(prompts, self.tokenized_prompts), dim=-1)
        prototypes, propagated = self.prompt_learner.apply_scgp(
            prompt_features, self.base_text_features, self.base_img_features,
            current_epoch=self.current_epoch)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ F.normalize(prototypes, dim=-1).t()
        if not self.training:
            return logits
        if label is None:
            raise ValueError('Training requires labels.')
        self.prompt_learner.update_confusion_matrix(logits.detach(), label)
        self.prompt_learner.update_class_hardness(logits.detach(), label)
        loss_ce = F.cross_entropy(logits, label)
        loss_sccm = sccm_loss(prompt_features, self.base_text_features) * self.cfg.TRAINER.ERRORCOMPASS.get('SCCM_LAMBDA', 1.0)
        teacher = self.teacher_logits(image_features.detach(), logit_scale.detach())
        loss_kl = kl_teacher_loss(logits, teacher) * self.cfg.TRAINER.ERRORCOMPASS.get('KL_LAMBDA', 1.0)
        loss_chc = self.prompt_learner.compute_chc_loss(logits, label, current_epoch=self.current_epoch) * self.prompt_learner.lambda_chc
        loss_cpmc = (self.prompt_learner.compute_cpmc_loss(propagated, current_epoch=self.current_epoch)
                     * self.prompt_learner.lambda_cpmc)
        return logits, loss_ce, loss_sccm, loss_kl, loss_chc, loss_cpmc


@TRAINER_REGISTRY.register()
class ErrorCompass_BiomedCLIP(CrossDatasetEvaluationMixin, TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ERRORCOMPASS.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")

        biomedclip_model, preprocess, _ = load_biomedclip_from_local(cfg.TRAINER.ERRORCOMPASS.CHECKPOINT_DIR)
        biomedclip_model.to(self.device)
        if cfg.TRAINER.ERRORCOMPASS.PREC == "fp32" or cfg.TRAINER.ERRORCOMPASS.PREC == "amp":
            biomedclip_model.float()

        print("Building ErrorCompass model")
        prototype_loader = None
        if str(cfg.DATASET.SUBSAMPLE_CLASSES).lower() != 'new':
            from dassl.data.data_manager import build_data_loader
            from dassl.data.transforms import build_transform
            prototype_loader = build_data_loader(
                cfg, sampler_type='SequentialSampler', data_source=self.dm.dataset.train_x,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=build_transform(cfg, is_train=False), is_train=False)
        self.model = ErrorCompassModel(
            cfg, classnames, biomedclip_model.eval(), prototype_loader
        )

        print("Turning off gradients in both the image and the text encoder")

        # Encoders and pretrained logit scale were frozen in model construction.
        # Default trainables are ctx, W and z; optional gate parameters remain available.
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            checkpoint = load_checkpoint(cfg.MODEL.INIT_WEIGHTS)
            state = checkpoint['state_dict']
            self._validate_algorithm_version(state)
            self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler(enabled=self.device.type == "cuda") if cfg.TRAINER.ERRORCOMPASS.PREC == "amp" else None


        # Keep one shared full-batch confusion state, matching the single-GPU protocol.
        # DataParallel would accumulate independent replica-local confusion counts.

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.current_epoch = self.epoch
        use_amp = self.cfg.TRAINER.ERRORCOMPASS.PREC == 'amp'
        with autocast(enabled=use_amp and image.is_cuda):
            logits, loss_ce, loss_sccm, loss_kl, loss_chc, loss_cpmc = self.model(image, label)
            loss = loss_ce + loss_sccm + loss_kl + loss_chc + loss_cpmc
        if use_amp:
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.model_backward_and_update(loss)
        summary = {'loss': loss.item(), 'acc': compute_accuracy(logits, label)[0].item(),
                   'loss_ce': loss_ce.item(), 'loss_sccm': loss_sccm.item(),
                   'loss_kl': loss_kl.item(), 'loss_chc': loss_chc.item(), 'loss_cpmc': loss_cpmc.item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    @staticmethod
    def _validate_algorithm_version(state):
        version = state.get('algorithm_version')
        if version is None or int(version.item()) != ALGORITHM_VERSION:
            raise RuntimeError('This checkpoint predates the paper-aligned ErrorCompass algorithm. '
                               f'Train version {ALGORITHM_VERSION} in a new output directory; old results are not interchangeable.')

    def resume_model_if_exist(self, directory):
        for name in self.get_model_names():
            folder = osp.join(directory, name)
            pointer = osp.join(folder, 'checkpoint')
            if osp.isfile(pointer):
                with open(pointer, encoding='utf-8') as handle:
                    filename = handle.readline().strip()
                state = load_checkpoint(osp.join(folder, filename))['state_dict']
                self._validate_algorithm_version(state)
        return super().resume_model_if_exist(directory)

    def load_model(self, directory, epoch=None):
        if not directory:
            raise ValueError('A checkpoint directory is required for evaluation.')
        epoch_to_load = self.cfg.OPTIM.MAX_EPOCH if epoch is None else epoch
        model_file = 'model.pth.tar-' + str(epoch_to_load)
        for name in self.get_model_names():
            model_path = osp.join(directory, name, model_file)
            checkpoint = load_checkpoint(model_path)
            state = dict(checkpoint['state_dict'])
            self._validate_algorithm_version(state)
            model = self._models[name]
            is_novel = str(self.cfg.DATASET.SUBSAMPLE_CLASSES).lower() == 'new'
            skipped = set()
            if is_novel:
                skipped = {'base_text_features', 'base_img_features',
                           'prompt_learner.token_prefix', 'prompt_learner.token_suffix',
                           'prompt_learner.confusion_matrix', 'prompt_learner.confusion_adjacency',
                           'prompt_learner.chc_step_counter', 'prompt_learner.class_reliability',
                           'prompt_learner.class_hardness', 'prompt_learner.hardness_count'}
                for key in skipped:
                    state.pop(key, None)
            incompatible = model.load_state_dict(state, strict=not is_novel)
            if is_novel and (set(incompatible.missing_keys) - skipped or incompatible.unexpected_keys):
                raise RuntimeError(f'Unexpected checkpoint incompatibility: {incompatible}')
            model.current_epoch = 0 if is_novel else max(0, int(checkpoint['epoch']) - 1)
            print(f'Loaded paper-aligned weights from {model_path}')

    def build_target_model(self, target_cfg, classnames, support_loader):
        """Use labeled target-training means, source weights and no target updates."""
        source = self.model.module if hasattr(self.model, 'module') else self.model
        if source.prompt_learner.ctx.dim() != 2:
            raise ValueError('Cross-dataset transfer requires shared prompts (CSC=False).')
        backbone = copy.deepcopy(source.text_encoder.model)
        backbone.to(device=self.device, dtype=source.dtype).eval()
        target = ErrorCompassModel(target_cfg, classnames, backbone, train_loader_x=support_loader, transfer=True)
        target.to(self.device)
        shared = {name: value.detach().clone() for name, value in source.named_parameters()
                  if name.startswith('prompt_learner.') and not name.startswith('prompt_learner.ZS_image_encoder.')}
        target_parameters = dict(target.named_parameters())
        if not set(shared) <= set(target_parameters):
            raise RuntimeError('Source and target have different trainable module structures.')
        with torch.no_grad():
            for name, value in shared.items():
                if target_parameters[name].shape != value.shape:
                    raise ValueError(f'Cannot transfer class-dependent parameter {name}.')
                target_parameters[name].copy_(value)
            for key in ('alpha_it', 'beta_it'):
                getattr(target.prompt_learner, key).copy_(getattr(source.prompt_learner, key))
            target.prompt_learner.confusion_matrix.zero_()
            target.prompt_learner.confusion_adjacency.zero_()
            target.prompt_learner.chc_step_counter.zero_()
        target.current_epoch = 0
        target.requires_grad_(False)
        target.eval()
        return target
