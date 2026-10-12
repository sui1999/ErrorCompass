
from __future__ import annotations
from typing import Any, Dict, Optional

_enabled: bool = False


def init(project: str, experiment: str = "", config: Optional[Dict[str, Any]] = None,
         offline: bool = False, work_dir: Optional[str] = None) -> None:
    """Initialize SwanLab. Safe to call multiple times; later calls are ignored.
    """
    global _enabled
    if _enabled:
        return
    try:
        import swanlab
        mode = "offline" if offline else "cloud"

        swanlab.init(project=project, experiment_name=experiment or None,
                     mode=mode, work_dir=work_dir, config=config)
        _enabled = True
    except Exception:

        _enabled = False


def enabled() -> bool:
    return _enabled


def finish() -> None:
    global _enabled
    if not _enabled:
        return
    try:
        import swanlab
        swanlab.finish()
    except Exception:
        pass
    _enabled = False


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if not _enabled:
        return
    try:
        import swanlab
        swanlab.log(metrics, step=step)
    except Exception:
        pass


def log_images(images: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log images where values are numpy arrays (H,W) or (H,W,C) or PIL.Image.
    They will be wrapped by swanlab.Image automatically.
    """
    if not _enabled:
        return
    try:
        import swanlab
        payload = {}
        for k, v in images.items():
            try:
                payload[k] = swanlab.Image(v)
            except Exception:

                continue
        if payload:
            swanlab.log(payload, step=step)
    except Exception:
        pass

