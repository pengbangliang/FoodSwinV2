from collections import defaultdict, deque
import torch.nn as nn
import time
import torch
import torch.distributed as dist
import logging
import errno
import os

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        a = a.cpu().flatten()
        b = b.cpu().flatten()

        #b = b.argmax(1).cpu().flatten()
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        # print(type(acc))
        # print(acc)
        new_acc=acc.clone()
        new_iu=iu.clone()
        for i in range(new_acc.size(0)):
            if torch.isnan(new_acc[i]):  # 检查是否为NaN
                new_acc[i] = 0
        for i in range(new_iu.size(0)):
            if torch.isnan(new_iu[i]):  # 检查是否为NaN
                new_iu[i] = 0
        return (
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'mean Accuracy: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}').format(
                acc_global.item() * 100,
                ['{:.2f}'.format(i) for i in (new_acc * 100).tolist()],
                new_acc.mean().item() * 100,
                ['{:.2f}'.format(i) for i in (new_iu * 100).tolist()],
                new_iu.mean().item() * 100)

def create_logger(logger_file_path):

    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class param_state():
    def __init__(self, model: nn.Module,
                 decay: float = 0.99):
        self.model = model
        self.decay = decay
        self.best = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.best[name] = param.data
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.best
                param.data = (1.0 - self.decay) * param.data + self.decay * self.best[name]

    def storage(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.best
                self.best[name] = param.data

