import torch
import torch.nn.functional as F

class SoftCropAugmentation:
    def __init__(self, n_class, sigma=0.3, k=2):
        self.chance = 1/n_class
        self.sigma = sigma
        self.k = k
    def draw_offset(self, limit, sigma=0.3, n=100):
        # draw an integer from a (clipped) 10 Gaussian
        for d in range(n):
            x = torch.randn((1))*sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)
    def __call__(self, image, label): 
        # typically, dim1 = dim2 = 32 for Cifar 18
        dim1, dim2 = image.size(1), image.size(2)
        # pad image
        image_padded = torch.zeros((3, dim1 * 3,dim2 * 3))
        image_padded[:, dim1:2*dim1, dim2:2*dim2] = image
        # draw tx, ty
        tx = self.draw_offset(dim1, self.sigma_crop * dim1)
        ty = self.draw_offset(dim2, self.sigma_crop * dim2)
        # crop image
        left, right = tx + dim1, tx + dim1 * 2
        top, bottom = ty + dim2, ty + dim2 * 2
        new_image = image_padded[:, left: right, top: bottom]
        # compute transformed image visibility and confidence
        v = (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)
        confidence = 1 - (1 - self.chance) * (1 - v) ** self.k
        return new_image, label, confidence

class SoftCropAugmentation_TS:
    def __init__(self, n_class, sigma=0.3, k=2):
        self.chance = 1/n_class
        self.sigma = sigma
        self.k = k
    def draw_offset(self, limit, sigma=0.3, n=100):
        # draw an integer from a (clipped) 10 Gaussian
        for d in range(n):
            x = torch.randn((1))*sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)
    def __call__(self, image, label): 
        # typically, dim1 = dim2 = 32 for Cifar 18
        seq_len , c = image.shape
        # pad image
        image_padded = torch.zeros((seq_len * 3,c))
        image_padded[seq_len:2*seq_len,:] = image
        # draw tx
        tx = self.draw_offset(seq_len, self.sigma_crop * seq_len)
        # crop sequence
        left, right = tx + seq_len, tx + seq_len * 2
        new_image = image_padded[left: right, :]
        # compute transformed image visibility and confidence
        v = (seq_len - abs(tx)) / seq_len
        confidence = 1 - (1 - self.chance) * (1 - v) ** self.k #confidence formula
        return new_image, label, confidence

class SoftConfidence(torch.nn.Module):
    def __init__(self, n_class, sigma=0.3, k=2):
        super().__init__()
        self.chance = 1/n_class
        self.sigma = sigma
        self.k = k
    def forward(self, v):
        confidence = 1 - (1 - self.chance) * (1 - v) ** self.k #confidence formula
        return confidence

class Soft_Criterion(torch.nn.Module):
    def __init__(self, confidence=0.9):
        super().__init__()
        self.confidence = confidence
        print('Using soft augment with confidence ',confidence)
    def forward(self,pred, label, confidence=None):
        if not torch.is_tensor(confidence):
            confidence = self.confidence
        log_prob = F.log_softmax(pred, dim=1)
        n_class = pred.size(1)
        # make soft one-hot target
        one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
        one_hot.scatter_(dim=1, index=label.view(-1,1), value= confidence)
        print('Softmax predict: ',log_prob) #!tmp
        print('Origin label: ',label.view(-1,1)) #!tmp
        print('Soften label: ',one_hot) #!tmp
        # compute weighted KL loss 10
        kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
        return kl.mean()

def soft_criterion(pred, label, confidence=0.9):
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1) #(bs,n_class)
    # make soft one-hot target
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    one_hot.scatter_(dim=1, index=label.view(-1,1), value= confidence)
    # compute weighted KL loss 10
    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1) #(bs)
    return kl