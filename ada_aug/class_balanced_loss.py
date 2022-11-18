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
from sklearn.utils.class_weight import compute_class_weight,\
    compute_sample_weight

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

def make_loss(multilabel,**kwargs):
    if not multilabel:
        return nn.CrossEntropyLoss(**kwargs)
    else:
        return nn.BCEWithLogitsLoss(reduction='mean',**kwargs)

def make_class_balance_count(train_labels,search_labels=np.array([]),multilabel=False,n_class=5):
    if not multilabel:
        train_labels_count = np.array([np.count_nonzero(train_labels == i) for i in range(n_class)]) #formulticlass
        search_labels_count = np.array([np.count_nonzero(search_labels == i) for i in range(n_class)]) #formulticlass
    else:
        train_labels_count = np.sum(train_labels,axis=0)
        search_labels_count = np.sum(search_labels,axis=0)
    tot_labels_count = train_labels_count + search_labels_count + 1 #smooth
    sim_type = 'softmax'
    if multilabel:
        sim_type = 'focal'
    print('Labels count: ',tot_labels_count)
    return ClassBalLoss(tot_labels_count,len(tot_labels_count),loss_type=sim_type).cuda()

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
def wass_loss(logits,targets,target_pair,class_output,sim_target=None,embed=False):
    loss = 0
    if torch.is_tensor(sim_target): #only for embed simliar class distance
        each_loss = wasserstein_loss(sim_target,logits) #smaller more different
        loss += each_loss
    else:
        if embed:
            soft_logits = logits
        else:
            soft_logits = logits.softmax(dim=1)
        pairs_label = target_pair[targets.detach().cpu()] #(batch, k)
        print(f'class {targets} pair with {pairs_label}') #!tmp
        for e_k in range(pairs_label.shape[1]):
            target_output = class_output[pairs_label[:,e_k].view(-1)].to(soft_logits.device) #(batch, n_class)
            print(target_output.shape) #!tmp
            each_loss = wasserstein_loss(target_output,soft_logits) #smaller more different
            loss += each_loss

    return loss
def confidence_loss(logits,targets,target_pair,class_output,sim_target=None):
    n_class = class_output.shape[0]
    soft_logits = logits.softmax(dim=1) #(batch, n_class)
    pairs_label = target_pair[targets.detach().cpu()].to(soft_logits.device) #(batch, k)
    mask = torch.zeros(*logits.size()).to(soft_logits.device) #(batch, n_class)
    mask.scatter_(1, pairs_label, 1)
    loss = 0
    loss = torch.mean(mask * soft_logits) * n_class
    '''for e_k in range(pairs_label.shape[1]):
        print(soft_logits[pairs_label[:,e_k].view(-1)].shape)
        loss += torch.mean(soft_logits[pairs_label[:,e_k].view(-1)]) * n_class'''
    
    return loss

class ClassDistLoss(torch.nn.Module):
    def __init__(self, distance_func='conf',loss_choose='conf',similar=False,init_k=3,lamda=1.0,num_classes=10,use_loss=True,
        noaug_target='se'):
        super().__init__()
        self.loss_target = 'output' #['output','embed']
        if '_' in distance_func:
            self.loss_target = distance_func.split('_')[0]
            self.distance_func = distance_func.split('_')[1]
        else:
            self.distance_func = distance_func
        if '_' in loss_choose:
            self.distance_func = distance_func.split('_')[1]
        else:
            self.distance_func = distance_func
        print('loss_target: ',self.loss_target)
        print('distance_func: ',self.distance_func)
        self.similar = similar
        self.class_output_mat = None
        self.classpair_dist = []
        self.class_pairs = None
        self.k = init_k
        self.fill_value = 1e6
        self.updated = False
        self.lamda = lamda
        self.num_classes = num_classes
        self.use_loss = use_loss
        #distance / weight init
        self.classpair_dist = np.ones((num_classes,num_classes))
        self.classweight_dist = np.ones((num_classes))
        self.class_embed_mat = None
        self.noaug_target = noaug_target

    def update_distance(self,class_output_mat): #(n_class,n_class)
        self.classpair_dist = []
        n_class = class_output_mat.shape[0]
        if self.distance_func=='kl':
            cuc_func = kl
        elif self.distance_func=='wass':
            cuc_func = wass
        elif self.distance_func=='conf':
            cuc_func = confidence
        else:
            print('Unknown distance_func ',self.distance_func)
            raise
        for c in range(n_class):
            if class_output_mat[c].sum()==0:
                line = np.full(n_class, self.fill_value)
            line = cuc_func(c,class_output_mat) # class c distance to other class
            self.classpair_dist.append(line)
        self.classpair_dist = np.array(self.classpair_dist)
        return self.classpair_dist
    #update embed
    def update_embed(self,class_embed_mat):
        self.class_embed_mat = np.array(class_embed_mat)
        return self.class_embed_mat
    #for loss weight
    def update_weight(self,class_output_mat):
        '''
        Add noaug regular weight for:
        (1) self output low perfromance (s) (2) easy be predict (e)
        '''
        classweight_dist = []
        n_class = class_output_mat.shape[0]
        for c in range(n_class):
            c_output = class_output_mat[c,c] #0~1
            c_been_output = (class_output_mat[:,c].sum() - c_output) + 1e-2 #0~1 + smooth
            c_noaug_weight = 0
            if 's' in self.noaug_target:
                c_noaug_weight += 1.0-c_output
            if 'e' in self.noaug_target:
                c_noaug_weight += c_been_output
            #c_noaug_weight = (1.0-c_output) + c_been_output
            classweight_dist.append(c_noaug_weight)
            print(f'Class {c} similar/noaug weight: c perfrom: {1.0-c_output}, c been output: {c_been_output}, total: {c_noaug_weight}')
        self.classweight_dist = np.array(classweight_dist)
        return self.classweight_dist
    def update_classpair(self,class_output_mat):
        n_class = class_output_mat.shape[0]
        self.update_distance(class_output_mat)
        self.update_weight(class_output_mat)
        self.class_output_mat = class_output_mat
        print('### updating class distance pair ###')
        #print(self.class_output_mat)
        #print(self.classpair_dist)
        print(self.classweight_dist)
        #min distance
        class_pairs = np.zeros((n_class,self.k))
        for c in range(n_class): #tmp use min distance
            pair_dists = np.minimum(self.classpair_dist[c,:], self.classpair_dist[:,c])
            sorted_dist = np.argsort(pair_dists)
            class_pairs[c] = sorted_dist[1:self.k+1] #top k
        self.class_pairs = torch.from_numpy(class_pairs).long()
        self.updated = True
        
    def forward(self, logits, targets, sim_targets=None):
        if not self.updated or not self.use_loss:
            return 0
        #loss type
        if self.loss_choose=='wass':
            loss_func = wass_loss
        elif self.loss_choose=='conf':
            loss_func = confidence_loss
        else:
            print('Unknown loss_choose ',self.loss_choose)
            raise
        #output / embedding as matrix
        if self.loss_target=='output':
            class_out_mat = self.class_output_mat
        elif self.loss_target=='embed':
            class_out_mat = self.class_embed_mat
        else:
            print('Unknown distance target choose ',self.loss_choose)
            raise
        #calculate
        classdist_loss = loss_func(logits,targets,self.class_pairs,class_out_mat) * self.lamda
        print('difference loss: ',classdist_loss) #!tmp
        if self.similar and torch.is_tensor(sim_targets):
            classdist_loss -= loss_func(logits,targets,self.class_pairs,class_out_mat,sim_target=sim_targets) * self.lamda
            print('similar loss: ',classdist_loss) #!tmp
        return classdist_loss

#train class weight loss
def compute_class_weights_dict(train_y, n_class): #!!!need change
    class_weights_dict = None
    smooth = np.arange(n_class)
    train_y = np.concatenate([train_y,smooth],axis=0)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_y),
        y=train_y
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    return class_weights_dict

def make_class_weights_samples(train_y, n_class, search_y=np.array([])):
    if len(search_y)>0:
        tot_y = np.concatenate([train_y,search_y],axis=0)
    else:
        tot_y = train_y
    class_dict = compute_class_weights_dict(tot_y,n_class)
    train_sample_w = compute_sample_weight(class_dict,train_y)
    if len(search_y)>0:
        search_sample_w = compute_sample_weight(class_dict,search_y)
        return train_sample_w,search_sample_w
    else:
        return train_sample_w

def make_class_weights(train_y, n_class, search_y=np.array([])):
    if len(search_y)>0:
        tot_y = np.concatenate([train_y,search_y],axis=0)
    else:
        tot_y = train_y
    class_dict = compute_class_weights_dict(tot_y,n_class)
    print('Class weights:')
    print(class_dict)
    return np.array([class_dict[i] for i in range(n_class)])

def make_class_weights_maxrel(train_y, n_class, search_y=np.array([]),multilabel=False):
    if len(search_y)>0:
        tot_y = np.concatenate([train_y,search_y],axis=0)
    else:
        tot_y = train_y
    if not multilabel:
        labels_count = np.array([np.count_nonzero(tot_y == i) for i in range(n_class)]) #formulticlass
    else:
        labels_count = np.sum(tot_y,axis=0)
    print('Class labels count:')
    print(labels_count)
    max_nclass = np.max(labels_count)
    class_weight = np.array([max_nclass / (count+1) for count in labels_count]) #with smooth
    print('Class weights:')
    print(class_weight)
    return class_weight

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
