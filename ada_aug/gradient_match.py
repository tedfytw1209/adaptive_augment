"""
Trying gradient match as Loss to update parameters
Functions from gradient match papers
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import grad

#### Hyperoptimization code from TaskAug ####
def zero_hypergrad(hyper_params):
    """
    :param get_hyper_train:
    :return:
    """
    current_index = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params

def store_hypergrad(hyper_params, total_d_val_loss_d_lambda):
    """

    :param get_hyper_train:
    :param total_d_val_loss_d_lambda:
    :return:
    """
    current_index = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params

def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner
    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter
        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, list(model.parameters()), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term
        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner

def get_hyper_train_flat(hyper_params): #flaaten all hyper params
    return torch.cat([p.view(-1) for p in hyper_params])

def gather_flat_grad(loss_grad): #flaaten all hyper params grad
    return torch.cat([p.reshape(-1) for p in loss_grad]) #g_vector

def get_loss(enc, x_batch_ecg, seqlen, y_batch, loss_obj,use_mix=False,multilabel=False):
    if use_mix:
        yhat = enc.classify(x_batch_ecg)
    else:
        yhat = enc.forward(x_batch_ecg, seqlen)
    if multilabel:
        y_batch = y_batch.float()
    else:
        y_batch = y_batch.long()
    loss = loss_obj(yhat.squeeze(), y_batch.squeeze())
    return loss

def do_aug(x,seqlen, y, aug,n_class,class_adaptive=False,multilabel=False,update_w=True):
    policy_y = None
    if class_adaptive: #target to onehot
        if not multilabel:
            policy_y = nn.functional.one_hot(y, num_classes=n_class).cuda().float()
        else:
            policy_y = y.cuda().float()
    mix_fea, _ = aug(x,seqlen, y=policy_y, mode='explore',update_w=update_w)
    return mix_fea

def hyper_step(model, aug, hyper_params, train_loader, optimizer, val_loader, elementary_lr, neum_steps, device, loss_obj,
        n_class,search_round=4,class_adaptive=False,multilabel=False,update_w=True):
    zero_hypergrad(hyper_params)
    num_weights = sum(p.numel() for p in model.parameters())
    d_train_loss_d_w = torch.zeros(num_weights).to(device)
    model.train(), model.zero_grad()
    input_search_list,seq_len_list,target_search_list,policy_y_list = [],[],[],[]
    aug_diff_loss,ori_search_loss = 0,0
    for batch_idx, (x, seqlen, y) in enumerate(train_loader):
        x = x.to(device).float()
        y = y.to(device)
        fea_x = do_aug(x,seqlen, y, aug,n_class,class_adaptive,multilabel,update_w)
        train_loss= get_loss(model, fea_x, seqlen, y, loss_obj,use_mix=True,multilabel=multilabel) / search_round
        optimizer.zero_grad()
        d_train_loss_d_w += gather_flat_grad(grad(train_loss, list(model.parameters()), 
                                                  create_graph=True, allow_unused=True))
        aug_diff_loss += train_loss.detach().mean().item()
        print('inner step loss: ',aug_diff_loss)
        if batch_idx==search_round-1: #not iterate all data
            break
    optimizer.zero_grad()
    # Initialize the preconditioner and counter
    # Compute gradients of the validation loss w.r.t. the weights/hypers
    d_val_loss_d_theta = torch.zeros(num_weights).cuda()
    model.train(), model.zero_grad()
    for batch_idx, (x, seqlen, y) in enumerate(val_loader):
        x = x.to(device).float()
        y = y.to(device)
        val_loss = get_loss(model, x,seqlen, y, loss_obj,multilabel=multilabel) / search_round
        ori_search_loss += val_loss.detach().mean().item()
        print('inner valid step loss: ',ori_search_loss)
        optimizer.zero_grad()
        d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters(), retain_graph=False))
        if class_adaptive: #target to onehot
            if not multilabel:
                policy_y = nn.functional.one_hot(y, num_classes=n_class).cuda().float()
            else:
                policy_y = y.cuda().float()
        policy_y_list.append(policy_y)
        input_search_list.append(x.detach())
        seq_len_list.append(seqlen.detach())
        target_search_list.append(y.detach())
        if batch_idx==search_round-1:
            break
    preconditioner = d_val_loss_d_theta
    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,neum_steps, model)
    indirect_grad = gather_flat_grad(grad(d_train_loss_d_w, hyper_params, grad_outputs=preconditioner.view(-1)))
    hypergrad = indirect_grad # + direct_Grad
    zero_hypergrad(hyper_params)
    store_hypergrad(hyper_params, -hypergrad)

    return hypergrad, aug_diff_loss, ori_search_loss, input_search_list,seq_len_list,target_search_list,policy_y_list