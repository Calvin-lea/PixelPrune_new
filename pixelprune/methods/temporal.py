"""
时序状态管理模块（AD-VLA 改进方案）。

提供两个无训练、低内存开销的时序组件：
    - TemporalScoreFusion：融合当前帧 saliency 与帧间 score_change 信号
    - ConsistencyBonus：对上帧保留的 patch 施加退出惯性，平滑 VLA 动作输出

设计原则：
    - 零训练参数，完全免训练
    - 内存代价极低（仅存上一帧的 scores [N float] 和 keep_indices [K int]）
    - 接口对称：fuse() 消费当前帧，update() 更新状态，reset() 跨序列重置

参考：
    AD_VLA_IMPROVEMENT.md — 改进三（score_change）、改动四（consistency_bonus）
"""

from __future__ import annotations

import torch


class TemporalScoreFusion:
    """
    时序 saliency 融合。

    将当前帧的预测误差 saliency（recon_score）与帧间正向变化量
    （score_change）加权融合，捕捉"新出现的重要目标"。

    score_change[i] = max(recon_score[i] - prev_scores[i], 0)
        → 上帧低分、这帧高分：新出现的目标，额外加权
        → 上帧高分、这帧低分：正常衰减，clamp 为 0，不惩罚

    Args:
        alpha: recon_score 权重，捕捉静态前景（纹理复杂区域）
        beta:  score_change 权重，捕捉动态/新出现目标

    内存开销：N × 4 bytes（单摄像头），N ≈ 800 时约 3.2 KB
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3) -> None:
        if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
            raise ValueError(f"alpha/beta 须在 [0, 1]，got alpha={alpha}, beta={beta}")
        self.alpha = alpha
        self.beta = beta
        self.prev_scores: torch.Tensor | None = None

    def fuse(self, recon_score: torch.Tensor) -> torch.Tensor:
        """
        融合当前帧 recon_score 与 score_change。

        Args:
            recon_score: 当前帧 patch 预测误差，shape [N]，值域 [0, 1]

        Returns:
            fused_score: 融合后的 saliency，shape [N]
        """
        if self.prev_scores is None:
            # 首帧：无上帧参考，直接返回 recon_score
            fused = recon_score
        else:
            if self.prev_scores.shape != recon_score.shape:
                # 序列长度变化（如分辨率切换），重置并退化为单帧
                self.prev_scores = None
                return recon_score

            score_change = (recon_score - self.prev_scores).clamp(min=0.0)
            fused = self.alpha * recon_score + self.beta * score_change

        # 更新状态（detach 避免误入计算图）
        self.prev_scores = recon_score.detach().clone()
        return fused

    def reset(self) -> None:
        """重置时序状态（换视频序列或摄像头时调用）。"""
        self.prev_scores = None

    def __repr__(self) -> str:
        return (
            f"TemporalScoreFusion(alpha={self.alpha}, beta={self.beta}, "
            f"has_prev={'yes' if self.prev_scores is not None else 'no'})"
        )


class ConsistencyBonus:
    """
    时序一致性退出惯性。

    对上一帧 top-K 保留的 patch 施加分数加成，防止微小的 fused_score
    波动导致 token 集合帧间跳变，从而平滑 VLA 动作预测序列。

    物理含义：
        gamma = 0.0  → 无惯性，每帧完全独立（原始 PixelPrune 行为）
        gamma = 0.15 → 上帧保留的 patch，这帧 score 需下降 > 0.15 才会被踢出
        gamma > 0.30 → 过度保守，对新出现目标响应变慢（不推荐）

    Args:
        gamma: 退出惯性强度，推荐范围 [0.10, 0.25]

    内存开销：K × 4 bytes（K 个 int32 索引），K ≈ 400 时约 1.6 KB
    """

    def __init__(self, gamma: float = 0.15) -> None:
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma 须在 [0, 1]，got {gamma}")
        self.gamma = gamma
        self.prev_keep: torch.Tensor | None = None

    def apply(self, scores: torch.Tensor) -> torch.Tensor:
        """
        给上帧保留的 patch 加 gamma bonus。

        Args:
            scores: 当前帧 fused_score，shape [N]

        Returns:
            scores_with_bonus: shape [N]，上帧 keep 位置加了 gamma
        """
        if self.prev_keep is None or self.gamma == 0.0:
            return scores

        bonus = torch.zeros_like(scores)
        # 过滤越界索引（防止序列长度变化时崩溃）
        valid = self.prev_keep[self.prev_keep < len(scores)]
        bonus[valid] = self.gamma
        return scores + bonus

    def update(self, keep_indices: torch.Tensor) -> None:
        """
        用本帧的 keep_indices 更新状态，供下一帧使用。

        Args:
            keep_indices: 本帧 top-K 选出的 patch 索引，shape [K]
        """
        self.prev_keep = keep_indices.detach().clone()

    def reset(self) -> None:
        """重置时序状态（换视频序列或摄像头时调用）。"""
        self.prev_keep = None

    def __repr__(self) -> str:
        k = len(self.prev_keep) if self.prev_keep is not None else 0
        return f"ConsistencyBonus(gamma={self.gamma}, prev_keep_size={k})"

