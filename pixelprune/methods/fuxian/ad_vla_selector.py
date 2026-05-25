"""
AD-VLA 时空 Token 选择器。

在 PixelPrune pred_2d 基础上叠加五个改动，实现训练-free 的时空 token 选择：

     改动一：连续 saliency score（预测误差 float，替代 bool 阈值）
     改动二·重构：P90 场景复杂度自适应 ratio
     改动三：时序信号融合（recon_score + score_change）
     改动四·重构：自适应 gamma（静止帧强惯性，运动帧弱惯性）
     改动五：空间多样性 top-K（4×4 网格硬约束）

使用方式：
    PIXELPRUNE_METHOD=ad_vla python your_script.py

    或：
    from pixelprune.methods import get_selector
    selector = get_selector("ad_vla", method="mae", ratio_min=0.15, ratio_max=0.6)

参考：AD_VLA_IMPROVEMENT.md
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .base import BasePatchSelector
from .pred_2d import Pred2DSelector
from .temporal import ConsistencyBonus, TemporalScoreFusion


def _spatial_diverse_topk(
    scores: torch.Tensor,
    k: int,
    patch_h: int,
    patch_w: int,
    grid_h: int = 4,
    grid_w: int = 4,
    min_per_cell: int = 2,
) -> torch.Tensor:
    """
    空间多样性 top-K 选择。

    两阶段：
      阶段1：每个 grid_h×grid_w 格子强制选 top-min_per_cell（硬约束）
      阶段2：剩余预算 k - 已选数量，按全局 score 从未选 patch 补充

    Args:
        scores:     fused_score（已加 consistency bonus），shape [N]
        k:          总保留数
        patch_h:    merged token 网格高度
        patch_w:    merged token 网格宽度
        grid_h:     网格行数
        grid_w:     网格列数
        min_per_cell: 每格最少保留 patch 数

    Returns:
        keep: 已排序的保留索引，shape [≤k]
    """
    device = scores.device
    cell_h = max(1, patch_h // grid_h)
    cell_w = max(1, patch_w // grid_w)
    mandatory_list: List[torch.Tensor] = []

    for i in range(grid_h):
        for j in range(grid_w):
            row_start = i * cell_h
            row_end = min((i + 1) * cell_h, patch_h)
            col_start = j * cell_w
            col_end = min((j + 1) * cell_w, patch_w)

            cell_mask = torch.zeros(patch_h, patch_w, dtype=torch.bool, device=device)
            cell_mask[row_start:row_end, col_start:col_end] = True
            cell_idx = cell_mask.view(-1).nonzero(as_tuple=True)[0]

            if len(cell_idx) == 0:
                continue
            n_pick = min(min_per_cell, len(cell_idx))
            top_in_cell = cell_idx[scores[cell_idx].topk(n_pick).indices]
            mandatory_list.append(top_in_cell)

    if mandatory_list:
        mandatory = torch.cat(mandatory_list).unique()
    else:
        mandatory = torch.empty(0, dtype=torch.long, device=device)

    remaining = max(0, k - len(mandatory))
    if remaining > 0:
        global_mask = torch.ones(len(scores), dtype=torch.bool, device=device)
        global_mask[mandatory] = False
        candidates = global_mask.nonzero(as_tuple=True)[0]
        if len(candidates) > 0:
            n_extra = min(remaining, len(candidates))
            extra = candidates[scores[candidates].topk(n_extra).indices]
            keep = torch.cat([mandatory, extra]).sort().values
        else:
            keep = mandatory.sort().values
    else:
        keep = mandatory[:k].sort().values

    return keep


class ADVLASelector(BasePatchSelector):
    """
    面向自动驾驶 VLA 的训练-free 时空 token 选择器。

    继承 BasePatchSelector，实现 select() 接口，可通过
    pixelprune 的注册表按名称 'ad_vla' 调用。

    Args:
        method:       距离度量（'mae'|'rmse'|'max'|'exact'），推荐 'mae'
        threshold:    pred_2d 的相似度阈值（仅用于 LOCO-I 边缘推断，不影响 top-K）
        ratio_min:    最低保留率（静止/均匀场景下限），∈ (0, 1)
        ratio_max:    最高保留率（复杂/分化场景上限），∈ (0, 1)
        safe_floor:   突发目标安全阀保留率托底
        gamma_max:    静止帧一致性惯性强度（uniqueness→0 时）
        gamma_min:    运动帧一致性惯性强度（uniqueness→1 时）
        alpha:        recon_score 在 fused_score 中的权重
        beta:         score_change 在 fused_score 中的权重
        grid_h:       空间多样性网格行数
        grid_w:       空间多样性网格列数
        min_per_cell: 每个网格格子的最少保留 patch 数
    """

    name = "ad_vla"
    aliases = ["advla", "ad_vla_selector"]

    def __init__(
        self,
        method: str = "mae",
        threshold: float = 0.0,
        ratio_min: float = 0.0,
        ratio_max: float = 0.6,
        safe_floor: float = 0.0,
        gamma_max: float = 0.15,
        gamma_min: float = 0.05,
        alpha: float = 0.7,
        beta: float = 0.3,
        grid_h: int = 4,
        grid_w: int = 4,
        min_per_cell: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(method, threshold, **kwargs)
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.safe_floor = safe_floor
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.min_per_cell = min_per_cell

        self._scorer = Pred2DSelector(method=method, threshold=threshold)
        self._temporal = TemporalScoreFusion(alpha=alpha, beta=beta)
        self._consistency = ConsistencyBonus(gamma=gamma_max)
        self._prev_merged: Optional[torch.Tensor] = None
        self._prev_recon_scores: Optional[torch.Tensor] = None

    def _pixel_uniqueness(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> float:
        return (current - previous).abs().mean().clamp(0.0, 1.0).item()

    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        indices_list: List[torch.Tensor] = []
        offset = 0

        for length, (t, h, w) in zip(merged_lengths, image_grid_thw.tolist()):
            img = merged_pv[offset: offset + length]
            mh = h // spatial_merge_size
            mw = w // spatial_merge_size
            device = img.device

            # 改动一：pred_2d 预测误差 → 连续 saliency score [N]
            recon_score = self._scorer.compute_saliency(img, mh, mw, device)

            # 边界处理：LOCO-I 无法预测第一行/列，原值 1.0 垄断 top-K
            # 改为用非边界中位数填充，使边界与内部 patch 公平竞争
            if mh > 1 and mw > 1:
                view_score = recon_score.view(mh, mw)
                inner_median = view_score[1:, 1:].quantile(0.5)
                view_score[0, :] = inner_median
                view_score[:, 0] = inner_median

            # 改动二·重构：saliency 分布自适应 ratio → K
            prev_slice = None
            if self._prev_merged is not None:
                if offset < self._prev_merged.shape[0]:
                    prev_slice = self._prev_merged[offset: offset + length]
                elif self._prev_merged.shape[0] >= length:
                    prev_slice = self._prev_merged[-length:]
            if prev_slice is not None and prev_slice.shape[0] == length:
                uniqueness = self._pixel_uniqueness(img, prev_slice)
            else:
                uniqueness = 0.5

            # P90 → 场景复杂度（高 P90 = 强纹理/边缘 → 多保留）
            complexity = min(1.0, recon_score.quantile(0.9).item())
            ratio = self.ratio_min + (self.ratio_max - self.ratio_min) * complexity

            # 安全阀：突发目标（score 大幅上升的 patch）→ 托底
            if self._prev_recon_scores is not None:
                score_change = (recon_score - self._prev_recon_scores).clamp(min=0)
                if score_change.max().item() > 0.3:
                    ratio = max(ratio, self.safe_floor)
            self._prev_recon_scores = recon_score.detach().clone()
            k = max(1, int(length * ratio))

            # 改动四·重构：自适应 gamma（静止帧强惯性，运动帧弱惯性）
            gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * uniqueness
            self._consistency.gamma = gamma

            # 改动三：时序信号融合（recon_score + score_change）
            fused = self._temporal.fuse(recon_score)

            # 改动四：时序一致性 bonus（使用自适应 gamma）
            fused = self._consistency.apply(fused)

            # 改动五：空间多样性 top-K
            keep = _spatial_diverse_topk(
                fused, k, mh, mw,
                self.grid_h, self.grid_w, self.min_per_cell,
            )

            self._consistency.update(keep)
            indices_list.append(keep)

            # 逐帧更新 _prev_merged，使后续帧能引用当前帧
            if self._prev_merged is None:
                self._prev_merged = img.detach().clone()
            else:
                end = offset + length
                if end > self._prev_merged.shape[0]:
                    pad = img.detach().clone()[:end - self._prev_merged.shape[0]]
                    self._prev_merged = torch.cat([self._prev_merged, pad])
                else:
                    self._prev_merged[offset:end] = img.detach().clone()
            offset += length
        return indices_list

    def reset(self) -> None:
        """重置所有时序状态（换视频序列时调用）。"""
        self._temporal.reset()
        self._consistency.reset()
        self._prev_merged = None
        self._prev_recon_scores = None

    def __repr__(self) -> str:
        return (
            f"ADVLASelector(method={self.method!r}, "
            f"ratio=[{self.ratio_min},{self.ratio_max}], "
            f"gamma=[{self.gamma_min},{self.gamma_max}], "
            f"alpha={self._temporal.alpha}, beta={self._temporal.beta}, "
            f"grid={self.grid_h}x{self.grid_w})"
        )