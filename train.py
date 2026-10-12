import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.lungcolon
import datasets.chmnist
import datasets.covid
import datasets.btmri
import datasets.ctkidney
import datasets.kvasir
import datasets.retina
import datasets.kneexray
import datasets.dermamnist 
import datasets.octmnist

import trainers.ErrorCompass.errorcompass_biomedclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)



def _default_prompt_cfg_values():
    return {
        "CTX_INIT": "a photo of a",
        "CSC": False,
        "CLASS_TOKEN_POSITION": "end",
        "N_CTX": 4,
        "PREC": "fp32",
        "N_PROMPTS": 50,
        "SCCM_LAMBDA": 1.0,
        "KL_LAMBDA": 1.0,
        "TEMPLATE_TAU": 1.5,
        "SCGP_NUM_LAYERS": 1,
        "USE_SCGP_ADAPTIVE_GATE": False,
        "SCGP_RELIABILITY_TEMP": 2.0,
        "SCGP_HARDNESS_TEMP": 1.0,
        "SCGP_ALPHA_BASE": 0.5,
        "SCGP_BETA_BASE": 0.3,
        "SCGP_GATE_EMA_RHO": 0.9,
        "CHC_TOP_K": 3,
        "CHC_TAU": 1.0,
        "CHC_BETA": 5.0,
        "CHC_ETA": 0.25,
        "LAMBDA_CHC": 0.5,
        "CONF_EMA_RHO": 0.9,
        "SCGP_SEMANTIC_WEIGHT": 0.3,
        "CONF_WARMUP_EPOCHS": 5,
        "CONF_UPDATE_FREQ": 10,
        "CPMC_MARGIN_BASE": 0.5,
        "CPMC_MARGIN_SCALE": 0.5,
        "CPMC_THRESHOLD": 0.2,
        "LAMBDA_CPMC": 0.3,
        "CPMC_TOP_K": 5,
        "CHECKPOINT_DIR": "checkpoints",
    }


def _copy_cfg_node_values(src, dst):
    for key in src.keys():
        dst[key] = src[key]



def _sync_prompt_nodes(left, right, preferred_side):
    defaults = _default_prompt_cfg_values()
    for key, default_value in defaults.items():
        left_val = left[key]
        right_val = right[key]
        left_changed = left_val != default_value
        right_changed = right_val != default_value

        if left_changed and not right_changed:
            right[key] = left_val
        elif right_changed and not left_changed:
            left[key] = right_val
        elif left_changed and right_changed and left_val != right_val:
            if preferred_side == "left":
                right[key] = left_val
            else:
                left[key] = right_val



def _set_promptmix_override(cfg, field_name, value):
    for namespace in ("BIOMEDCOOP", "ERRORCOMPASS", "TRICLIP"):
        if hasattr(cfg.TRAINER, namespace):
            setattr(getattr(cfg.TRAINER, namespace), field_name, value)



def _sync_prompt_trainer_cfg(cfg, args):
    trainer_name = cfg.TRAINER.NAME or ""
    opt_items = [str(item) for item in getattr(args, "opts", [])]
    has_errorcompass_opts = any(item.startswith("TRAINER.ERRORCOMPASS.") for item in opt_items)
    has_biomed_opts = any(item.startswith("TRAINER.BIOMEDCOOP.") for item in opt_items)
    config_file = args.config_file or ""

    errorcompass_node = cfg.TRAINER.ERRORCOMPASS
    biomed_node = cfg.TRAINER.BIOMEDCOOP

    if trainer_name.startswith("ErrorCompass_"):
        preferred = "left"
        if has_biomed_opts and not has_errorcompass_opts:
            preferred = "right"
        elif has_errorcompass_opts:
            preferred = "left"
        elif "configs/trainers/ErrorCompass/" in config_file:
            preferred = "left"
        elif "configs/trainers/BiomedCoOp/" in config_file:
            preferred = "right"
        _sync_prompt_nodes(errorcompass_node, biomed_node, preferred)
    elif trainer_name.startswith("BiomedCoOp_"):
        preferred = "right"
        if has_errorcompass_opts and not has_biomed_opts:
            preferred = "left"
        _sync_prompt_nodes(errorcompass_node, biomed_node, preferred)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed is not None and args.seed >= 0:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if any(getattr(args, field, None) is not None for field in
           ('promptmix_min_dropout', 'promptmix_max_dropout', 'pmix_ma_momentum')):
        raise ValueError('PromptMix is not part of the manuscript; remove the PM options.')



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 4
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4
    cfg.TRAINER.COCOOP.CSC = False
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp32"
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"
    def init_errorcompass_cfg(node):
        for key, value in _default_prompt_cfg_values().items():
            node[key] = value

    cfg.TRAINER.BIOMEDCOOP = CN()
    init_errorcompass_cfg(cfg.TRAINER.BIOMEDCOOP)

    cfg.TRAINER.ERRORCOMPASS = CN()
    init_errorcompass_cfg(cfg.TRAINER.ERRORCOMPASS)

    cfg.TRAINER.TRICLIP = CN()
    init_errorcompass_cfg(cfg.TRAINER.TRICLIP)
    cfg.TRAINER.TRICLIP.PROMPTMIX_ENABLED = True
    cfg.TRAINER.TRICLIP.PROMPTMIX_DROPOUT = 0.3
    cfg.TRAINER.TRICLIP.PROMPTMIX_LAMBDA = 0.1
    cfg.TRAINER.TRICLIP.PROMPTMIX_POWER = 1.0
    cfg.TRAINER.TRICLIP.DIFF_ATT_MODE = "both"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)


    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)


    if args.config_file:
        cfg.merge_from_file(args.config_file)


    reset_cfg(cfg, args)


    cfg.merge_from_list(args.opts)

    _sync_prompt_trainer_cfg(cfg, args)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    print("Trainer built successfully.")

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    elif not args.no_train:
        trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="a nonnegative value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--promptmix-min-dropout", type=float, default=None, help="min dropout for promptmix")
    parser.add_argument("--promptmix-max-dropout", type=float, default=None, help="max dropout for promptmix")
    parser.add_argument("--pmix-ma-momentum", type=float, default=None, help="moving average momentum for promptmix")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
