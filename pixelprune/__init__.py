"""
PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding.

为 Vision-Language Models 提供基于像素预测编码的 visual token 压缩，
在 ViT 编码器之前通过空间冗余检测移除重复 patch，实现高效推理与训练。

内置扫描策略:
    - raster:     行主序扫描，对相邻 token 连续去重
    - serpentine: 蛇形扫描，同时捕获水平与垂直方向重复
    - pred_2d:    LOCO-I 2D 预测编码（默认推荐），利用三个因果邻居推断边缘方向

快速使用::

    # HuggingFace 后端：通过 monkey-patch 注入（在加载模型前调用）
    import os
    os.environ["PIXELPRUNE_ENABLED"] = "true"
    os.environ["PIXELPRUNE_METHOD"] = "pred_2d"  # 默认

    from pixelprune import apply_pixelprune
    apply_pixelprune(model="qwen3_vl")   # 或 "qwen3_5"

    # vLLM 后端：不需要调用 apply_pixelprune —— 由 setup.py 注册的
    # ``vllm.general_plugins`` entry point 在每个 vLLM 进程启动时自动加载。
    # 用户只需设置 PIXELPRUNE_ENABLED=true 后正常使用 vLLM 即可。

    # 也可以直接调用选择器
    from pixelprune import compute_merged_keep_indices
    indices_list = compute_merged_keep_indices(pixel_values, image_grid_thw)

扩展自定义方法::

    from pixelprune.methods import register_method, BasePatchSelector

    @register_method
    class MySelector(BasePatchSelector):
        name = "my_method"
        def select(self, pixel_values, image_grid_thw, spatial_merge_size=2):
            ...
"""

# --- 核心调度 & 索引工具 ---
from .core import (
    compute_merged_keep_indices,
    merged_indices_to_patch_indices,
)

# --- 方法注册表（供扩展自定义方法） ---
from .methods import (
    BasePatchSelector,
    register_method,
)


__version__ = "1.0.0"

__all__ = [
    "compute_merged_keep_indices",
    "merged_indices_to_patch_indices",
    "apply_pixelprune",
    "BasePatchSelector",
    "register_method",
]


def apply_pixelprune(model: str = "qwen3_vl") -> None:
    """对 Qwen3-VL / Qwen3.5 的 HuggingFace 实现应用 monkey-patch。

    需在加载模型前调用。**vLLM 后端无需调用此函数** —— PixelPrune 通过
    ``vllm.general_plugins`` entry point 在每个 vLLM 进程启动时自动加载。

    Args:
        model: 模型架构，'qwen3_vl' 或 'qwen3_5'。
    """
    from .patches import apply_patches as _apply
    _apply(model=model)
