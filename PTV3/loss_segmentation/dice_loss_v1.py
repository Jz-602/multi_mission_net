import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Binary Dice loss on a whole point set for one semantic class."""
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        if predict.numel() != target.numel():
            raise ValueError("predict and target must have the same number of elements")

        ## predict/target: [num_valid_points], 表示某一语义类在所有有效点上的预测概率与 one-hot 标签。
        predict = predict.reshape(-1)
        target = target.reshape(-1).type_as(predict)

        numerator = 2 * torch.sum(predict * target) + self.smooth
        denominator = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth
        loss = 1 - numerator / denominator

        if self.reduction == 'mean':
            return loss
        if self.reduction == 'sum':
            return loss
        if self.reduction == 'none':
            return loss

        raise ValueError(f'Unexpected reduction: {self.reduction}')


class DiceLossV1(nn.Module):
    """Multi-class Dice loss for point-level semantic logits.

    Args:
        ignore_index: class index to ignore
        predict: A tensor of shape [num_valid_points, num_classes]
        target: A tensor of shape [num_valid_points]
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, ignore_index, smooth=1, exponent=2):
        super(DiceLossV1, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.exponent = exponent

    def forward(self, input, target):
        mask = (target != self.ignore_index)
        target = target[mask]
        input = input[mask]

        ## input: [num_valid_points, num_classes]，是点级 logits。
        ## target: [num_valid_points]，是每个有效点的语义类别索引。
        if mask.sum() == 0:
            return input.sum() * 0.0

        dice = BinaryDiceLoss(self.smooth, self.exponent)
        input = F.softmax(input, dim=1)
        target = F.one_hot(target.long(), num_classes=input.shape[1]).type_as(input)

        total_loss = input.new_tensor(0.0)
        valid_class_count = 0

        ## 只在当前 batch 实际出现过的类别上求 Dice，避免大量空类别把监督冲淡。
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                if torch.sum(target[:, i]) == 0:
                    continue
                total_loss += dice(input[:, i], target[:, i])
                valid_class_count += 1

        if valid_class_count == 0:
            return input.sum() * 0.0

        return total_loss / valid_class_count
