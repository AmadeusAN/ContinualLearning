import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import autocast


class SelfDistillationLoss(nn.Module):
    def __init__(
        self, model, config, distill_dim=256, temperature=2.0, lambda_distill=0.4
    ):
        super().__init__()
        self.model = model  # 只在 __init__ 时引用一下，用于访问 projectors
        self.config = config
        self.distill_dim = distill_dim
        self.temperature = temperature
        self.lambda_distill = lambda_distill

        # 如果你希望 projector 完全独立，也可以在这里重新定义，但推荐直接用 model 里的

    def forward(self, output_dict):
        enc_feats = output_dict["enc_features"]
        dec_feats = output_dict["dec_features"]
        logits = output_dict["logits"]

        loss_dist = 0.0

        with autocast(device_type="cuda", enabled=self.config.use_amp):
            # ==================== Encoder 侧 ====================
            for i, student in enumerate(enc_feats[:-1]):
                teacher = enc_feats[-1]
                student_proj = self.model.enc_projectors[i](student)
                teacher_proj = self.model.enc_projectors[-1](teacher)

                if student_proj.shape[2:] != teacher_proj.shape[2:]:
                    teacher_proj = F.interpolate(
                        teacher_proj,
                        size=student_proj.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                loss_dist += F.mse_loss(student_proj, teacher_proj.detach())

            # ==================== Decoder 侧 ====================
            for i, student in enumerate(dec_feats[1:]):  # 从第2个开始（较浅层）
                teacher = dec_feats[0]
                student_proj = self.model.dec_projectors[i + 1](student)
                teacher_proj = self.model.dec_projectors[0](teacher)

                if student_proj.shape[2:] != teacher_proj.shape[2:]:
                    teacher_proj = F.interpolate(
                        teacher_proj,
                        size=student_proj.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                loss_dist += F.mse_loss(student_proj, teacher_proj.detach())

            # ==================== Prediction 侧 KL ====================
            soft_logits = F.log_softmax(logits / self.temperature, dim=1)
            soft_target = F.softmax(logits.detach() / self.temperature, dim=1)
            loss_kl = F.kl_div(soft_logits, soft_target, reduction="batchmean") * (
                self.temperature**2
            )

            total_dist_loss = self.lambda_distill * (loss_dist + loss_kl)
            return total_dist_loss


class BinaryDiceLoss(nn.Module):
    """计算单个二值通道的 Dice Loss"""

    def __init__(self, smooth=1.0):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        predict: (N, *) 已经过 sigmoid 的概率值 [0, 1]
        target:  (N, *) 二值标签 {0, 1}
        """
        assert predict.shape == target.shape, "predict & target shape don't match"

        predict = predict.contiguous().view(predict.shape[0], -1)  # (N, HW)
        target = target.contiguous().view(target.shape[0], -1)  # (N, HW)

        intersection = torch.sum(predict * target, dim=1)  # (N,)
        denominator = (
            torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth
        )  # (N,)

        dice_score = (2.0 * intersection + self.smooth) / denominator
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, organ_list):
        total_loss = []
        predict = F.sigmoid(predict)

        total_loss = []
        B = predict.shape[0]

        if target.shape[1] == 1:
            # 需要进行 one-hot 编码
            target = F.one_hot(target.long().squeeze(1), num_classes=predict.shape[1])
            target = target.permute(0, 4, 1, 2, 3).contiguous()

        for b in range(B):
            for organ in organ_list:
                # organ - 1，而你的 organ_list 是从 1 开始的，看样子是完全不需要预测背景了，BTCV 的完整 organ_list 是 1 2 3 4 5 6 7 8 9 10 11 12
                # 注意，对应 target 的话，是不需要减 1 的
                dice_loss = self.dice(predict[b, organ - 1], target[b, organ])
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)

        return total_loss.sum() / total_loss.shape[0]


class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, organ_list):
        assert predict.shape[2:] == target.shape[2:], (
            "predict & target shape do not match"
        )
        total_loss = []
        B = predict.shape[0]
        if target.shape[1] == 1:
            # 需要进行 one-hot 编码
            target = F.one_hot(target.long().squeeze(1), num_classes=predict.shape[1])
            target = target.permute(0, 4, 1, 2, 3).contiguous().float()

        for b in range(B):
            for organ in organ_list:
                ce_loss = self.criterion(predict[b, organ - 1], target[b, organ])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        return total_loss.sum() / total_loss.shape[0]


class Multi_MSELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_MSELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.MSELoss()

    def forward(self, predict, target, organ_list, last_stage_organ_list):
        assert predict.shape[2:] == target.shape[2:], (
            "predict & target shape do not match"
        )
        predict = F.sigmoid(predict)
        target = F.sigmoid(target)
        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            for organ in last_stage_organ_list:
                # 注意这里用来和 Logits 做 mse，它们的通道是一一对应的
                ce_loss = self.criterion(predict[b, organ - 1], target[b, organ - 1])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)
        return total_loss.sum() / total_loss.shape[0]
