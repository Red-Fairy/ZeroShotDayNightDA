import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import autograd, nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step -
                                                             warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr

class InvariancePenaltyLoss(nn.Module):
    r"""Invariance Penalty Loss from `Invariant Risk Minimization <https://arxiv.org/pdf/1907.02893.pdf>`_.
    We adopt implementation from `DomainBed <https://github.com/facebookresearch/DomainBed>`_. Given classifier
    output :math:`y` and ground truth :math:`labels`, we split :math:`y` into two parts :math:`y_1, y_2`, corresponding
    labels are :math:`labels_1, labels_2`. Next we calculate cross entropy loss with respect to a dummy classifier
    :math:`w`, resulting in :math:`grad_1, grad_2` . Invariance penalty is then :math:`grad_1*grad_2`.
    Inputs:
        - y: predictions from model
        - labels: ground truth
    Shape:
        - y: :math:`(N, C)` where C means the number of classes.
        - labels: :math:`(N, )` where N mean mini-batch size
    """

    def __init__(self):
        super(InvariancePenaltyLoss, self).__init__()
        self.scale = torch.tensor(1.).requires_grad_()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_1 = F.cross_entropy(y[::2] * self.scale, labels[::2])
        loss_2 = F.cross_entropy(y[1::2] * self.scale, labels[1::2])
        grad_1 = autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        return penalty