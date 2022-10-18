import torch
import torch.nn.functional as F


def non_saturating_loss(logits, targets):
    probs = logits.softmax(1)
    log_prob = torch.log(1 - probs + 1e-12)
    if targets.ndim == 2: #if it is onehot
        return - (targets * log_prob).sum(1).mean()
    else:
        return F.nll_loss(log_prob, targets)


class NonSaturatingLoss(torch.nn.Module):
    def __init__(self, epsilon=0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if self.epsilon > 0: # label smoothing
            n_classes = logits.shape[1]
            onehot_targets = F.one_hot(targets, n_classes).float()
            targets = (1 - self.epsilon) * onehot_targets + self.epsilon / n_classes
        return non_saturating_loss(logits, targets)

class Wasserstein_loss(torch.nn.Module):
    def __init__(self, reduction=None):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        ndim = len(y_pred.shape)
        if self.reduction==None:
            out = torch.mean(y_true * y_pred,dim=[i for i in range(1,ndim)])
        elif self.reduction=='mean':
            out = torch.mean(y_true * y_pred)
        elif self.reduction=='sum':
            out = torch.sum(torch.mean(y_true * y_pred,dim=[i for i in range(1,ndim)]))
        return out