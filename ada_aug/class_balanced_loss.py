"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""
import numpy as np
import torch
import torch.nn.functional as F

"""Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import torch.nn as nn
from scipy.special import kl_div
from scipy.stats import wasserstein_distance

def sigmoid(x):
    return 1/(1 + np.exp(-x))

"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
""" 
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).long()
        ones = - torch.ones(input.shape).cuda()
        for i in range(input.shape[0]):
            ones[i, target[i,0].item()] = 1 
        input = input * ones
        #logpt = F.log_softmax(input)
        logpt = F.logsigmoid(input)
        #logpt = logpt.gather(1,target)
        #logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        #print(pt)
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.sum(1)
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            #logpt = logpt * Variable(at)
            loss = loss * Variable(at)

        #loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, np.maximum(samples_per_cls,1.0))
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)
    print('CB loss weights:')
    print(weights)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

class ClassBalLoss(torch.nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type='softmax', beta=0.9999, gamma=2.0):
        super().__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        return CB_loss(targets, logits, self.samples_per_cls, self.no_of_classes, self.loss_type, self.beta, self.gamma)

'''def create_diff_loss(class_difficulty=None, loss_params=('dynamic', None, True)):
    print('Using CDB Softmax Loss')
    tau, focal_gamma, normalize = loss_params
    
    if class_difficulty is not None:
        epsilon = 0.01
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/ (1 - np.max(class_difficulty) + epsilon) - 1
            tau = 2 * sigmoid(bias)
        else:
            tau = float(tau)
        
        cdb_weights = class_difficulty ** tau
        if normalize:
           cdb_weights = (cdb_weights / cdb_weights.sum()) * len(cdb_weights)      
        if focal_gamma is not None:
             return FocalLoss(gamma=float(focal_gamma), alpha=torch.FloatTensor(cdb_weights))
        else:
             return nn.CrossEntropyLoss(weight=torch.FloatTensor(cdb_weights),)
    else:
        sys.exit('Class Difficulty can not be None')'''

class ClassDiffLoss(torch.nn.Module):
    def __init__(self, class_difficulty=np.array([]), tau='dynamic', focal_gamma=None, normalize=True, epsilon=0.01):
        super().__init__()
        self.class_difficulty = class_difficulty
        self.tau = tau
        self.focal_gamma = focal_gamma
        self.normalize = normalize
        self.loss_func = None
        self.epsilon = epsilon
        if len(class_difficulty)>0:
            self.update_weight(class_difficulty)
        else:
            if self.focal_gamma is not None:
                self.loss_func = FocalLoss().cuda()
            else:
                self.loss_func = nn.CrossEntropyLoss().cuda()

    def update_weight(self,class_difficulty):
        if self.tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/ (1 - np.max(class_difficulty) + self.epsilon) - 1
            tau = 2 * sigmoid(bias)
        else:
            tau = float(self.tau)
        cdb_weights = class_difficulty ** tau
        if self.normalize:
            cdb_weights = (cdb_weights / cdb_weights.sum()) * len(cdb_weights)
        
        if self.focal_gamma is not None:
            self.loss_func = FocalLoss(gamma=float(self.focal_gamma), alpha=torch.FloatTensor(cdb_weights)).cuda()
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(cdb_weights),).cuda()
        self.class_difficulty = class_difficulty
    def forward(self, logits, targets):
        return self.loss_func(logits, targets)

def kl(c,class_output):
    n_class = class_output.shape[0]
    out_arr = np.zeros(n_class)
    for k in range(n_class):
        if c==k: continue
        dist = kl_div(class_output[c],class_output[k])+1e-9
        out_arr[k] = dist
    return out_arr

def wass(c,class_output):
    n_class = class_output.shape[0]
    out_arr = np.zeros(n_class)
    for k in range(n_class):
        if c==k: continue
        dist = wasserstein_distance(class_output[c],class_output[k])+1e-9 #need weight???
        out_arr[k] = dist
    return out_arr

def confidence(c,class_output):
    n_class = class_output.shape[0]
    out_arr = np.zeros(n_class)
    c_output = class_output[c]
    for k in range(n_class):
        if c==k: continue
        dist = max(c_output[c] - c_output[k],0)+1e-9 #clip
        out_arr[k] = dist
    return out_arr

def wasserstein_loss(y_true, y_pred): 
    #smaller when two distribution different, minus if need make distribution similar
    return torch.mean(y_true * y_pred)
def wass_loss(logits,targets,target_pair,class_output):
    soft_logits = logits.softmax(dim=1)
    pairs_label = target_pair[targets] #(batch, k)
    print(pairs_label.shape)
    loss = 0
    for e_k in range(pairs_label.shape[1]):
        target_output = class_output[pairs_label[:,e_k].view(-1)].detach() #(batch, n_class)
        each_loss = wasserstein_loss(target_output,soft_logits) #smaller more different
        loss += each_loss

    return loss
def confidence_loss(logits,targets,target_pair,class_output):
    soft_logits = logits.softmax(dim=1)
    pairs_label = target_pair[targets] #(batch, k)
    print(pairs_label.shape)
    loss = torch.mean(soft_logits[pairs_label])
    return loss

class ClassDistLoss(torch.nn.Module):
    def __init__(self, distance_func='conf',loss_choose='conf',init_k=3):
        super().__init__()
        self.distance_func = distance_func
        self.loss_choose = loss_choose
        self.class_output_mat = None
        self.classpair_dist = []
        self.class_pairs = None
        self.k = init_k
        self.fill_value = 1e6

    def update_distance(self,class_output_mat): #(n_class,n_class)
        self.classpair_dist = []
        n_class = class_output_mat.shape[0]
        if self.distance_func=='kl':
            cuc_func = kl
        elif self.distance_func=='wass':
            cuc_func = wass
        elif self.distance_func=='conf':
            cuc_func = confidence
        for c in range(n_class):
            if class_output_mat[c].sum()==0:
                line = np.full(n_class, self.fill_value)
            line = cuc_func(c,class_output_mat) # class c distance to other class
            self.classpair_dist.append(line)
        self.classpair_dist = np.array(self.classpair_dist)
        return self.classpair_dist
    
    def update_classpair(self,class_output_mat):
        self.update_distance(class_output_mat)
        self.class_output_mat = class_output_mat
        #min distance
        class_pairs = np.zeros((n_class,self.k))
        n_class = class_output_mat.shape[0]
        for c in n_class: #tmp use min distance
            pair_dists = np.minimum(self.classpair_dist[c,:], self.classpair_dist[:,c])
            sorted_dist = np.argsort(pair_dists)
            class_pairs[c] = sorted_dist[1:self.k+1] #top k
        self.class_pairs = torch.from_numpy(class_pairs).long()
        
    def forward(self, logits, targets):
        if self.distance_func=='wass':
            loss_func = wass_loss
        elif self.distance_func=='conf':
            loss_func = confidence_loss
        classdist_loss = loss_func(logits,targets,self.class_pairs,self.class_output_mat)
        return classdist_loss

if __name__ == '__main__':
    no_of_classes = 5
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "focal"
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    print(cb_loss)
