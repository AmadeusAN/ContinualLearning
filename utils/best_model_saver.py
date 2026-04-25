import torch.nn as nn
import torch
from os.path import join


class model_param_saver:
    def __init__(self, save_dir: str = None, monitor_metric: str = None):
        self.save_dir = save_dir
        self.monitor_metric = monitor_metric
        self.high_score = 0.0

    def check(self, model: nn.Module, metrics=None):
        """在一次验证之后, 将与实验相关的所有参数都保存以来，以便下次继续训练

        Args:
            model (_type_): _description_
            optimizer (_type_): 某些优化器内部也保存了权重的信息，如一阶矩和二阶矩
            scheduler (_type_): 某些学习率调度器内部也保存了权重的信息
            metrics (_type_): 当时的度量，用于判断保存最佳模型
            epoch_idx (_type_): 当时的epoch
        """
        if (
            metrics is not None
            and metrics["mean"][self.monitor_metric] > self.high_score
        ):
            print("检测到更高表现，替换 best_model")
            self.high_score = metrics["mean"][self.monitor_metric]
            torch.save(
                (model.state_dict()),
                join(self.save_dir, "best_model.pth"),
            )
