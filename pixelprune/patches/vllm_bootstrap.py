from __future__ import annotations

import os


def _enabled() -> bool:
    return os.environ.get("PIXELPRUNE_ENABLED", "").lower() in ("true", "1", "yes")


def maybe_apply_patches() -> None:
    """vLLM 调用的入口函数。未启用时立即返回，启用时才懒加载真正的 patch。"""
    if not _enabled():
        return

    # Qwen3.5 的 apply_patches 会同时 patch 共享父类 Qwen3-VL；
    # 若 vllm 版本不含 qwen3_5 模型，则退回到只 patch Qwen3-VL。
    try:
        from .qwen3_5_vllm import apply_patches
    except ImportError:
        try:
            from .qwen3_vl_vllm import apply_patches
        except ImportError:
            return

    try:
        apply_patches()
    except Exception as e:
        try:
            from vllm.logger import init_logger
            init_logger(__name__).warning("PixelPrune vLLM plugin failed: %s", e)
        except Exception:
            import logging
            logging.getLogger(__name__).warning("PixelPrune vLLM plugin failed: %s", e)
