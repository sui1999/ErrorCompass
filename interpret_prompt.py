import os
import argparse
import torch

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer


def find_ctx_in_state(state_dict: dict):
    """Try to find the saved context tensor inside a checkpoint state_dict.
    Common key is 'prompt_learner.ctx'. Return the tensor or raise KeyError.
    """

    if "prompt_learner.ctx" in state_dict:
        return state_dict["prompt_learner.ctx"]

    for k in state_dict.keys():
        if k.endswith("prompt_learner.ctx") or k.endswith("prompt_learner.ctx.weight"):
            return state_dict[k]

    for k in state_dict.keys():
        if k.endswith(".ctx") and "prompt" in k:
            return state_dict[k]
    raise KeyError("Could not find prompt_learner.ctx in checkpoint state_dict")


def decode_token(tokenizer, token_idx: int):
    """Decode single token id to string using tokenizer object returned by get_tokenizer.
    The tokenizer has a `.tokenizer.decode()` method in this repo's setup.
    """
    try:

        return tokenizer.tokenizer.decode(int(token_idx))
    except Exception:
        try:
            return tokenizer.tokenizer.decode([int(token_idx)])
        except Exception:
            return str(int(token_idx))


def main():
    parser = argparse.ArgumentParser(description="Interpret learned prompt context tokens: list nearest vocabulary words and distances.")
    parser.add_argument("--fpath", type=str, required=True, help="Path to the learned prompt checkpoint (model.pth.tar or similar)")
    parser.add_argument("--topk", default=5, type=int, help="Number of nearest words to show per context token")
    parser.add_argument("--force-full-load", action="store_true", help="If set, skip weights-only safe load and load checkpoint fully (unsafe; only use for trusted files)")
    parser.add_argument("--allow-globals", type=str, default=None, help="Comma-separated list of fully-qualified global names to allowlist for weights-only load, e.g. 'torch.optim.lr_scheduler.CosineAnnealingLR'")
    args = parser.parse_args()

    fpath = args.fpath
    topk = args.topk
    force_full = args.force_full_load
    allowed_globs_cli = args.allow_globals
    user_allowed = None
    if allowed_globs_cli:
        user_allowed = [g.strip() for g in allowed_globs_cli.split(',') if g.strip()]

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Checkpoint not found: {fpath}")

    print(f"Loading BiomedCLIP model and tokenizer (this may take a moment)...")
    biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    biomedclip_model = biomedclip_model.float().eval()

    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    token_embedding = biomedclip_model.text.transformer.embeddings.word_embeddings.weight.detach().cpu()

    print(f"Loading checkpoint: {fpath}")







    import re
    import importlib
    import warnings

    ckpt = None
    tried_weights_only = False

    def _torch_load(path, map_location, weights_only_flag=None):

        if weights_only_flag is None:
            return torch.load(path, map_location=map_location)
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only_flag)
        except TypeError:

            return torch.load(path, map_location=map_location)

    try:

        if force_full:
            print("--force-full-load specified: performing full checkpoint load (unsafe).")
            ckpt = _torch_load(fpath, map_location="cpu", weights_only_flag=False)
        else:

            ckpt = _torch_load(fpath, map_location="cpu", weights_only_flag=True)
            tried_weights_only = True
    except Exception as e:
        err_str = str(e)

        globals_found = re.findall(r"GLOBAL\s+([A-Za-z0-9_.]+)", err_str)
        if globals_found:
            print("Weights-only load failed due to unallowed GLOBALs:", globals_found)
            ser = getattr(torch, 'serialization', None)
            allowed = False
            if ser is not None:

                add_safe = getattr(ser, 'add_safe_globals', None)
                safe_ctx = getattr(ser, 'safe_globals', None)

                resolved = []

                if user_allowed:
                    globals_found = list(dict.fromkeys(globals_found + user_allowed))
                for g in globals_found:
                    try:
                        mod_name, attr_name = g.rsplit('.', 1)
                        mod = importlib.import_module(mod_name)
                        obj = getattr(mod, attr_name)
                        resolved.append(obj)
                    except Exception:
                        warnings.warn(f"Could not resolve global name to object: {g}; skipping allowlist for this name.")
                if len(resolved) == 0:

                    resolved = None
                if add_safe is not None:
                    try:
                        if resolved is not None:
                            with add_safe(resolved):
                                ckpt = _torch_load(fpath, map_location="cpu", weights_only_flag=True)
                                allowed = True
                        else:
                            allowed = False
                    except Exception:
                        allowed = False
                elif safe_ctx is not None:
                    try:
                        if resolved is not None:
                            with safe_ctx(resolved):
                                ckpt = _torch_load(fpath, map_location="cpu", weights_only_flag=True)
                                allowed = True
                        else:
                            allowed = False
                    except Exception:
                        allowed = False
            if not allowed:
                print("Could not allowlist unsafe globals via torch.serialization; falling back to full load (weights_only=False).")

                ckpt = _torch_load(fpath, map_location="cpu", weights_only_flag=False)
        else:

            print("Weights-only load failed and no unsafe GLOBALs detected; falling back to full load (weights_only=False).")
            ckpt = _torch_load(fpath, map_location="cpu", weights_only_flag=False)


    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    ctx = find_ctx_in_state(state_dict)
    ctx = ctx.detach().cpu().float()


    if ctx.dim() == 2:

        n_ctx = ctx.shape[0]
        distances = torch.cdist(ctx, token_embedding, p=2)
        sorted_idxs = torch.argsort(distances, dim=1)

        for i in range(n_ctx):
            idxs = sorted_idxs[i, :topk]
            words = [decode_token(tokenizer, idx.item()) for idx in idxs]
            dists = [distances[i, idx].item() for idx in idxs]
            entries = [f"{w} ({dist:.4f})" for w, dist in zip(words, dists)]
            print(f"Context token {i}: " + ", ".join(entries))

    elif ctx.dim() == 3:

        n_cls, n_ctx, _ = ctx.shape
        distances = torch.cdist(ctx.view(-1, ctx.shape[-1]), token_embedding, p=2)
        sorted_idxs = torch.argsort(distances, dim=1)
        for cls in range(n_cls):
            print(f"Class {cls}:")
            for j in range(n_ctx):
                idx_base = cls * n_ctx + j
                idxs = sorted_idxs[idx_base, :topk]
                words = [decode_token(tokenizer, idx.item()) for idx in idxs]
                dists = [distances[idx_base, idx].item() for idx in idxs]
                entries = [f"{w} ({dist:.4f})" for w, dist in zip(words, dists)]
                print(f"  Context token {j}: " + ", ".join(entries))
    else:
        raise ValueError(f"Unexpected ctx tensor dim: {ctx.dim()}")


if __name__ == '__main__':
    main()


