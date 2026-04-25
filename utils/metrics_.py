import torch
import torch.nn.functional as F
import numpy as np
from medpy.metric.binary import dc, jc, hd, asd

# from torchmetrics.functional.segmentation import hausdorff_distance   不支持 3D 数据
from collections import defaultdict
from medpy.metric.binary import dc, jc, hd95, asd, sensitivity, specificity


class SegmentationMetrics:
    """
    用于验证集的分割指标计算，支持按 organ_list 逐通道计算。

    支持的指标:
        - Dice Coefficient (DC)
        - Jaccard Coefficient (JC) / IoU
        - 95% Hausdorff Distance (HD95)
        - Average Surface Distance (ASD)
        - Sensitivity (Recall)
        - Specificity
    """

    def __init__(
        self, num_classes: int, organ_names: dict = None, metrics: list = None
    ):
        """
        Args:
            num_classes: 类别数（不含背景），对应 predict 的通道数
            organ_names: 器官编号到名称的映射，如 {1: 'spleen', 2: 'right_kidney', ...}
            metrics: 需要计算的指标列表，默认 ['dc', 'jc', 'hd95']
        """
        self.num_classes = num_classes
        self.organ_names = organ_names or {
            i: f"organ_{i}" for i in range(1, num_classes + 1)
        }
        self.metrics = metrics or ["dc", "jc", "hd95"]

        # 指标函数映射（均来自 medpy，输入为二值 numpy 数组）
        self.metric_fns = {
            "dc": dc,  # Dice Coefficient
            "jc": jc,  # Jaccard Coefficient (IoU)
            "hd95": hd95,  # 95% Hausdorff Distance
            "asd": asd,  # Average Surface Distance
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

        self.reset()

    def reset(self):
        """每个 epoch 开始前调用，清空累积结果"""
        # {metric_name: {organ_id: [values...]}}
        self.results = {m: defaultdict(list) for m in self.metrics}
        self.count = 0

    @torch.no_grad()
    def update(
        self,
        predict: torch.Tensor,
        target: torch.Tensor,
        organ_list: list,
        spacing: tuple = None,
        threshold: float = 0.5,
    ):
        """
        更新一个 batch 的指标。

        Args:
            predict: (B, C, D, H, W) 或 (B, C, H, W)，模型原始输出（logits）
            target:  (B, 1, D, H, W) 或 (B, C, D, H, W)，标签
            organ_list: 需要计算的器官编号列表，如 [1, 2, 3, ..., 12]（从1开始）
            spacing: 体素间距，用于距离类指标（如 hd95, asd），如 (1.0, 1.0, 1.0)
            threshold: sigmoid 后的二值化阈值
        """
        predict = torch.sigmoid(predict)
        B = predict.shape[0]

        # one-hot 编码 target（如果需要）
        if target.shape[1] == 1:
            target = F.one_hot(target.long().squeeze(1), num_classes=predict.shape[1])
            # (B, D, H, W, C) -> (B, C, D, H, W) 或 (B, H, W, C) -> (B, C, H, W)
            if target.dim() == 5:
                target = target.permute(0, 4, 1, 2, 3).contiguous()
            else:
                target = target.permute(0, 3, 1, 2).contiguous()

        # 二值化预测
        predict_bin = (predict > threshold).float()

        for b in range(B):
            for organ in organ_list:
                # organ 从 1 开始，通道索引 organ - 1，但是 target 的通道索引不变
                pred_np = predict_bin[b, organ - 1].cpu().numpy().astype(np.uint8)
                tgt_np = target[b, organ].cpu().numpy().astype(np.uint8)

                for metric_name in self.metrics:
                    fn = self.metric_fns[metric_name]
                    value = self._safe_compute(
                        fn, pred_np, tgt_np, metric_name, spacing
                    )
                    self.results[metric_name][organ].append(value)

            self.count += 1

    def _safe_compute(
        self,
        fn,
        pred: np.ndarray,
        tgt: np.ndarray,
        metric_name: str,
        spacing: tuple = None,
    ) -> float:
        """
        安全地计算指标，处理全零的边界情况。

        对于 Dice/JC：
            - pred 和 tgt 都为空 -> 1.0（完美预测"无该器官"）
            - 只有一方为空 -> 0.0
        对于距离类指标（hd95, asd）：
            - 任意一方为空 -> 返回 NaN（后续统计时忽略）
        """
        pred_empty = pred.sum() == 0
        tgt_empty = tgt.sum() == 0

        # 重叠类指标
        if metric_name in ["dc", "jc", "sensitivity", "specificity"]:
            if pred_empty and tgt_empty:
                return 1.0  # 两者都正确地预测为空
            if pred_empty or tgt_empty:
                if metric_name == "specificity" and pred_empty and not tgt_empty:
                    # specificity 可能仍然可计算，但通常返回 0
                    pass
                if metric_name in ["dc", "jc", "sensitivity"]:
                    if pred_empty and tgt_empty:
                        return 1.0
                    return 0.0
            try:
                return float(fn(pred, tgt))
            except Exception:
                return 0.0

        # 距离类指标
        elif metric_name in ["hd95", "asd"]:
            if pred_empty or tgt_empty:
                return float("nan")  # 无法计算距离
            try:
                if spacing is not None:
                    return float(fn(pred, tgt, voxelspacing=spacing))
                else:
                    return float(fn(pred, tgt))
            except Exception:
                return float("nan")

        # 其他
        else:
            try:
                return float(fn(pred, tgt))
            except Exception:
                return float("nan")

    def compute(self) -> dict:
        """
        计算所有累积样本的最终指标。

        Returns:
            {
                'per_organ': {
                    organ_id: {metric_name: mean_value, ...},
                    ...
                },
                'mean': {metric_name: overall_mean_value, ...},
                'organ_names': {organ_id: organ_name, ...}
            }
        """
        summary = {
            "per_organ": defaultdict(dict),
            "mean": {},
            "organ_names": self.organ_names,
        }

        all_metric_means = defaultdict(list)

        for metric_name in self.metrics:
            for organ_id, values in self.results[metric_name].items():
                # 过滤掉 NaN
                valid_values = [v for v in values if not np.isnan(v)]
                if len(valid_values) > 0:
                    organ_mean = np.mean(valid_values)
                else:
                    organ_mean = float("nan")

                organ_name = self.organ_names.get(organ_id, f"organ_{organ_id}")
                summary["per_organ"][organ_id][metric_name] = organ_mean
                summary["per_organ"][organ_id]["name"] = organ_name

                if not np.isnan(organ_mean):
                    all_metric_means[metric_name].append(organ_mean)

        # 计算所有器官的平均值
        for metric_name in self.metrics:
            values = all_metric_means.get(metric_name, [])
            summary["mean"][metric_name] = (
                np.mean(values) if len(values) > 0 else float("nan")
            )

        return dict(summary)

    def print_summary(self, summary: dict = None):
        """格式化打印指标"""
        if summary is None:
            summary = self.compute()

        print("\n" + "=" * 80)
        print(f"{'Organ':<20}", end="")
        for m in self.metrics:
            print(f"{m.upper():>12}", end="")
        print()
        print("-" * 80)

        for organ_id in sorted(summary["per_organ"].keys()):
            organ_info = summary["per_organ"][organ_id]
            name = organ_info.get("name", f"organ_{organ_id}")
            print(f"{name:<20}", end="")
            for m in self.metrics:
                val = organ_info.get(m, float("nan"))
                if np.isnan(val):
                    print(f"{'N/A':>12}", end="")
                elif m in ["hd95", "asd"]:
                    print(f"{val:>12.2f}", end="")
                else:
                    print(f"{val:>12.4f}", end="")
            print()

        print("-" * 80)
        print(f"{'MEAN':<20}", end="")
        for m in self.metrics:
            val = summary["mean"].get(m, float("nan"))
            if np.isnan(val):
                print(f"{'N/A':>12}", end="")
            elif m in ["hd95", "asd"]:
                print(f"{val:>12.2f}", end="")
            else:
                print(f"{val:>12.4f}", end="")
        print()
        print("=" * 80 + "\n")
