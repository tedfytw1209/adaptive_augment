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

def soft_target(pred, label, confidence):
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)
    # make soft one-hot target
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    one_hot.scatter_(dim=1, index=label, src= confidence)
    # compute weighted KL loss 10
    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
    return kl.mean()