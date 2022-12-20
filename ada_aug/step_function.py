import os
from pickle import FALSE
import sys
import time
import torch
import utils
import logging
import argparse
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
import torch.nn as nn
import torch.utils

from adaptive_augmentor import AdaAug,AdaAug_TS
from networks import get_model,get_model_tseries
from networks.projection import Projection_TSeries
from dataset import get_num_class, get_dataloaders, get_label_name, get_dataset_dimension, get_ts_dataloaders, get_num_channel
from config import get_warmup_config
from warmup_scheduler import GradualWarmupScheduler
import wandb
from gradient_match import hyper_step
from utils import mixup_data,mixup_aug

def train(args, train_queue, model, criterion, optimizer,scheduler, epoch, grad_clip, adaaug, multilabel=False,n_class=10,
        difficult_aug=False,reweight=True,lambda_aug = 1.0,class_adaptive=False,map_select=False,visualize=False,training=True,
        teach_rew=None,policy_apply=True,extra_criterions=[],noaug_reg='',mixup=False,mixup_alpha=1.0,aug_mix=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    tr_class_augw = torch.zeros(n_class,adaaug.n_ops).float()
    tr_class_augm = torch.zeros(n_class,adaaug.n_ops).float()
    confusion_matrix = torch.zeros(n_class,n_class)
    tr_output_matrix = torch.zeros(n_class,n_class).float()
    tr_count = torch.zeros(n_class).float()
    softmax_m = nn.Softmax(dim=1)
    preds = []
    targets = []
    total = 0
    if training:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
    #start training
    for step, (input, seq_len, target) in enumerate(train_queue):
        input = input.float().cuda()
        target = target.cuda(non_blocking=True)
        #  get augmented training data from adaaug
        policy_y = None
        if class_adaptive: #target to onehot
            if not multilabel:
                policy_y = nn.functional.one_hot(target, num_classes=n_class).cuda().float()
            else:
                policy_y = target.cuda().float()
        aug_images,_ = adaaug(input, seq_len, mode='exploit',y=policy_y,policy_apply=policy_apply)
        #mixup if need
        if mixup:
            aug_images, target_a, target_b, mixup_lam = mixup_data(aug_images,target,mixup_alpha)
        if aug_mix:
            aug_images = mixup_aug(aug_images,input,mixup_alpha)
        #start train
        if training:
            model.train()
        else:
            model.eval()
        optimizer.zero_grad()
        logits = model(aug_images, seq_len)
        '''if multilabel:
            aug_loss = criterion(logits, target.float())
        else:
            aug_loss = criterion(logits, target.long())'''
        #if mixup
        if mixup:
            aug_loss = mixup_lam * cuc_loss(logits,target_a,criterion,multilabel) + (1.0-mixup_lam) * cuc_loss(logits,target_b,criterion,multilabel)
            print('mixup: ',mixup_lam)
        else:
            aug_loss = cuc_loss(logits,target,criterion,multilabel)
        #difficult aug
        batch_size = target.shape[0]
        ori_loss = 0
        if difficult_aug:
            origin_logits = model(input, seq_len)
            if multilabel:
                ori_loss = criterion(origin_logits, target.float()) / 2
            else:
                ori_loss = criterion(origin_logits, target.long()) / 2
            if reweight: #reweight part, a,b = ?
                p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target].detach()
                p_aug = logits.softmax(dim=1)[torch.arange(batch_size), target].clone().detach()
                w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
                if w_aug.sum() > 0:
                    w_aug /= (w_aug.mean().detach() + 1e-6)
                else:
                    w_aug = 1
                aug_loss = (w_aug * lambda_aug * aug_loss).mean() / 2
            else:
                aug_loss = (lambda_aug * aug_loss).mean() / 2
        #teach reweight
        if teach_rew!=None:
            teach_ori_logits = model(input, seq_len)
            p_aug = logits.softmax(dim=1)[torch.arange(batch_size), target].clone().detach()
            t_aug = teach_ori_logits.softmax(dim=1)[torch.arange(batch_size), target].clone().detach()
            w_aug = torch.sqrt(t_aug * torch.clamp(t_aug - p_aug, min=0)) + 1 #a=0.5,b=0.5 ???
            w_aug /= (w_aug.mean().detach() + 1e-9)
            aug_loss = (w_aug * aug_loss).mean()
        loss = ori_loss + aug_loss.mean() #assert
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
        torch.cuda.empty_cache()
        n = input.size(0)
        objs.update(loss.detach().item(), n)
        if not multilabel:
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))    
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)

        global_step = step + epoch * len(train_queue)
        if global_step % args.report_freq == 0 and not multilabel:
            logging.info('train: step=%03d loss=%e top1acc=%f top5acc=%f', global_step, objs.avg, top1.avg, top5.avg)

        # log the policy
        policy = adaaug.add_history(input, seq_len, target,y=policy_y)
        tr_class_augm += policy[0] #sum of policy
        tr_class_augw += policy[1] #sum of policy
        # Accuracy / AUROC
        if not multilabel:
            _, predicted = torch.max(logits.data, 1)
            soft_out = softmax_m(logits).detach().cpu()
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            for i,t in enumerate(target.data.view(-1)):
                tr_output_matrix[t.long(),:] += soft_out[i]
                tr_count[t.long()] += 1
        else:
            predicted = torch.sigmoid(logits.data)
        preds.append(predicted.detach().cpu())
        targets.append(target.detach().cpu())
    #visualize
    if visualize:
        input_search, seq_len, target_search = next(iter(train_queue))
        input_search = input_search.float().cuda()
        target_search = target_search.cuda()
        policy_y = None
        if class_adaptive: #target to onehot
            if not multilabel:
                policy_y = nn.functional.one_hot(target_search, num_classes=n_class).cuda().float()
            else:
                policy_y = target_search.cuda().float()
        adaaug.visualize_result(input_search, seq_len,policy_y,target_search)
        exit()
    #table dic
    table_dic = {}
    tr_add_up = torch.clamp(tr_output_matrix.sum(dim=1,keepdim=True),min=1e-9)
    tr_count = torch.clamp(tr_count,min=1e-9).view(-1,1)
    table_dic['train_output'] = (tr_output_matrix / tr_add_up)
    table_dic['train_confusion'] = confusion_matrix
    targets_np = torch.cat(targets).numpy()
    preds_np = torch.cat(preds).numpy()
    table_dic['train_target'] = targets_np
    table_dic['train_predict'] = preds_np
    #tmp only look noaug %
    #print('sum tr output matrix: ',tr_add_up)
    tr_class_augm = tr_class_augm / tr_count #every output add up is one
    tr_class_augw = tr_class_augw / tr_count
    noaug_precent = tr_class_augw[:,0].view(-1) #only noaug %
    for i,e_c in enumerate(noaug_precent):
        table_dic[f'train_c{i}_id'] = e_c
    #class-wise & Total
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/torch.clamp(confusion_matrix.sum(1),min=1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / torch.clamp(confusion_matrix.sum(),min=1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
        ptype = 'acc'
        logging.info(f'Epoch train: loss={objs.avg} top1acc={top1.avg} top5acc={top5.avg}')
    else:
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom_cw2 = utils.mAP_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom_cw2.mean()
        logging.info('Epoch train: loss=%e macroAUROC=%f', objs.avg, perfrom)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Epoch train: loss=%e macroAP=%f', objs.avg, perfrom2)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw2])+']')
        ptype = 'auroc'
    #wandb dic
    out_dic = {}
    out_dic[f'train_loss'] = objs.avg
    out_dic[f'train_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'train_{ptype}_c{i}'] = e_c
    if multilabel: #addition
        add_type = 'ap'
        out_dic[f'train_{add_type}_avg'] = perfrom2
        for i,e_c in enumerate(perfrom_cw2):
            out_dic[f'train_{add_type}_c{i}'] = e_c
        if map_select:
            return perfrom2, objs.avg, out_dic, table_dic
    
    return perfrom, objs.avg, out_dic, table_dic

def infer(valid_queue, model, criterion, multilabel=False, n_class=10,mode='test',map_select=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    confusion_matrix = torch.zeros(n_class,n_class)
    output_matrix = torch.zeros(n_class,n_class).float()
    softmax_m = nn.Softmax(dim=1)
    preds = []
    targets = []
    preds_score = []
    targets_score = []
    total = 0
    with torch.no_grad():
        for input,seq_len, target in valid_queue:
            input = input.float().cuda()
            target = target.cuda(non_blocking=True)
            logits = model(input,seq_len)
            if multilabel:
                loss = criterion(logits, target.float())
            else:
                loss = criterion(logits, target.long())
            
            n = input.size(0)
            objs.update(loss.detach().item(), n)
            if not multilabel:
                _, predicted = torch.max(logits.data, 1)
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                top1.update(prec1.detach().item(), n)
                top5.update(prec5.detach().item(), n)
                soft_out = softmax_m(logits).detach().cpu()#(bs,n_class)
                for t, p in zip(target.data.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                for i,t in enumerate(target.data.view(-1)):
                    output_matrix[t.long(),:] += soft_out[i]
                total += target.size(0)
                each_pscore = torch.gather(soft_out,1,predicted.cpu().detach().long().view(-1,1))
                each_tscore = torch.gather(soft_out,1,target.cpu().detach().long().view(-1,1))
                preds_score.append(each_pscore) #(bs,n_class)[(bs)]=>(bs)
                targets_score.append(each_tscore)
            else:
                predicted = torch.sigmoid(logits.data)
                preds_score.append(predicted) #multilabel don't need
                targets_score.append(predicted) #multilabel don't need
            preds.append(predicted.cpu().detach())
            targets.append(target.cpu().detach())
    #table dic
    table_dic = {}
    table_dic[f'{mode}_output'] = (output_matrix / torch.clamp(output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic[f'{mode}_confusion'] = confusion_matrix
    #class-wise
    targets_np = torch.cat(targets).numpy()
    preds_np = torch.cat(preds).numpy()
    targets_score_np = torch.cat(targets_score).numpy()
    preds_score_np = torch.cat(preds_score).numpy()
    table_dic[f'{mode}_target'] = targets_np
    table_dic[f'{mode}_predict'] = preds_np
    table_dic[f'{mode}_target_score'] = targets_score_np
    table_dic[f'{mode}_predict_score'] = preds_score_np
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/torch.clamp(confusion_matrix.sum(1),min=1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / torch.clamp(confusion_matrix.sum(),min=1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
        perfrom2 = top5.avg
        ptype = 'acc'
    else:
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom_cw2 = utils.mAP_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom_cw2.mean()
        logging.info('Epoch %s: loss=%e macroAUROC=%f', mode, objs.avg, perfrom)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Epoch %s: loss=%e macroAP=%f',mode, objs.avg, perfrom2)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw2])+']')
        ptype = 'auroc'
    #wandb dic
    out_dic = {}
    out_dic[f'{mode}_loss'] = objs.avg
    out_dic[f'{mode}_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'{mode}_{ptype}_c{i}'] = e_c
    if multilabel: #addition
        add_type = 'ap'
        out_dic[f'{mode}_{add_type}_avg'] = perfrom2
        for i,e_c in enumerate(perfrom_cw2):
            out_dic[f'{mode}_{add_type}_c{i}'] = e_c
        if map_select:
            return perfrom2, objs.avg, perfrom2, objs.avg, out_dic, table_dic
    
    return perfrom, objs.avg, perfrom2, objs.avg, out_dic, table_dic

def sub_loss(ori_loss, aug_loss, lambda_aug,**kwargs): #will become to small
    return lambda_aug * (aug_loss - ori_loss.detach())
def relative_loss(ori_loss, aug_loss, lambda_aug, add_number=0,multilabel=False,**kwargs):
    out = lambda_aug * ((ori_loss.detach()+add_number) / (aug_loss+add_number))
    if multilabel:
        out = out.mean(1)
    return out
def relative_loss_mean(ori_loss, aug_loss, lambda_aug, add_number=0,**kwargs):
    aug_loss = aug_loss.mean()
    ori_loss = ori_loss.mean()
    return lambda_aug * ((ori_loss.detach()+add_number) / (aug_loss+add_number))
def relative_loss_s(ori_loss, aug_loss, lambda_aug, add_number=0,multilabel=False,**kwargs):
    out = lambda_aug * (2 * (ori_loss.detach()+add_number) / (aug_loss+ori_loss.detach()+add_number))
    if multilabel:
        out = out.mean(1)
    return out
def minus_loss(ori_loss, aug_loss, lambda_aug,multilabel=False,**kwargs):
    out = -1 * lambda_aug * aug_loss
    if multilabel:
        out = out.mean(1)
    return out
def adv_loss(ori_loss, aug_loss, lambda_aug,**kwargs):
    return lambda_aug * aug_loss
def none_loss(ori_loss, aug_loss, lambda_aug,**kwargs):
    return ori_loss

def ab_loss(ori_loss, aug_loss,multilabel=False,**kwargs):
    if multilabel:
        aug_loss = aug_loss.mean(1)
    return aug_loss
def rel_loss_s(ori_loss, aug_loss, add_number=0,multilabel=False,**kwargs):
    out = (2 * (aug_loss+add_number) / (ori_loss.detach()+add_number+aug_loss.detach()))
    if multilabel:
        out = out.mean(1)
    return out
def rel_loss(ori_loss, aug_loss, add_number=0,multilabel=False,**kwargs):
    out = ((aug_loss+add_number) / (ori_loss.detach()+add_number))
    if multilabel:
        out = out.mean(1)
    return out
def rel_loss_mean(ori_loss, aug_loss, add_number=0,**kwargs):
    aug_loss = aug_loss.mean()
    ori_loss = ori_loss.mean()
    return ((aug_loss+add_number) / (ori_loss.detach()+add_number))

def cuc_loss(logits,target,criterion,multilabel,**kwargs):
    if multilabel:
        loss = criterion(logits, target.float(),**kwargs)
    else:
        loss = criterion(logits, target.long(),**kwargs)
    return loss

def embed_mix(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel):
    aug_logits = gf_model.classify(mixed_features) 
    aug_loss = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
    return aug_loss, aug_logits, aug_loss

def output_mix(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel):
    # batch, n_ops(sub), n_hidden
    print(mixed_features.shape)
    # mixed_features=(batch, keep_lens, n_ops, n_hidden) or (batch, keep_lens, n_hidden) or (batch, n_ops, n_hidden)
    if len(aug_weights[1].shape)==3: # batch, keep_lens, n_ops, n_hidden
        batch, keep_lens,n_ops, n_hidden = mixed_features.shape
        weights, keeplen_ws = aug_weights[0], aug_weights[2]
        mixed_features = mixed_features.reshape(batch*n_ops*keep_lens,n_hidden)
        aug_logits = gf_model.classify(mixed_features)
        aug_logits = aug_logits.reshape(batch, keep_lens, n_ops,-1)
        aug_logits = [w.matmul(feat) for w, feat in zip(weights, aug_logits)] #[(keep_lens)]
        aug_logits = torch.stack([len_w.matmul(feat) for len_w,feat in zip(keeplen_ws,aug_logits)], dim=0) #[(1)]
    else:
        batch, n_param, n_hidden = mixed_features.shape
        weights = aug_weights[0]
        mixed_features = mixed_features.reshape(batch*n_param,n_hidden)
        aug_logits = gf_model.classify(mixed_features)
        aug_logits = aug_logits.reshape(batch, n_param,-1)
        aug_logits = torch.stack([w.matmul(feat) for w, feat in zip(weights, aug_logits)], dim=0) #[(1)]

    aug_loss = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
    return aug_loss, aug_logits

def embed_diff(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel):
    aug_logits = gf_model.classify(mixed_features) 
    aug_loss = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
    return aug_loss, aug_logits, aug_loss

def loss_mix(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel):
    # mixed_features=(batch, keep_lens, n_ops, n_hidden) or (batch, keep_lens, n_hidden) or (batch, n_ops, n_hidden)
    target_trsearch = torch.unsqueeze(target_trsearch,dim=1) #(bs,1) !!!tmp not for multilabel
    if len(aug_weights[1].shape)==3: ### !!! maybe have bug !!!
        batch, keep_lens,n_ops, n_hidden = mixed_features.shape #(batch, keep_lens, n_ops, n_hidden)
        weights, keeplen_ws = aug_weights[0], aug_weights[2]
        target_trsearch = target_trsearch.expand(-1,n_ops*keep_lens).reshape(batch*n_ops*keep_lens)
        mixed_features = mixed_features.reshape(batch*n_ops*keep_lens,n_hidden)
        aug_logits = gf_model.classify(mixed_features)
        aug_loss_all = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
        aug_loss_all = aug_loss_all.reshape(batch, keep_lens, n_ops)
        # weights = (bs,n_ops), aug_loss = (batch,keep_lens,n_ops)
        aug_loss = [w.matmul(feat) for w, feat in zip(weights, aug_loss_all)] #[(keep_lens)]
        aug_loss = torch.stack([len_w.matmul(feat) for len_w,feat in zip(keeplen_ws,aug_loss)], dim=0) #[(1)]
        #print('Loss mix aug_loss: ',aug_loss.shape,aug_loss)
        aug_loss = aug_loss.mean()
        aug_logits = aug_logits.reshape(batch, keep_lens, n_ops,-1)
    else:
        batch, n_param, n_hidden = mixed_features.shape
        weights = aug_weights[0]
        target_trsearch = target_trsearch.expand(-1,n_param).reshape(batch*n_param)
        mixed_features = mixed_features.reshape(batch*n_param,n_hidden)
        aug_logits = gf_model.classify(mixed_features)
        aug_loss_all = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
        aug_loss_all = aug_loss_all.reshape(batch, n_param)
        aug_loss = torch.stack([w.matmul(feat) for w, feat in zip(weights, aug_loss_all)], dim=0) #[(1)]
        #aug_loss = aug_loss.mean()
        aug_logits = aug_logits.reshape(batch, n_param,-1)
    #print('Loss mix aug_logits: ',aug_logits.shape,aug_logits)
    return aug_loss, aug_logits, aug_loss_all
#loss select
def loss_select(loss_type,adv_criterion,multilabel=False):
    #diff loss criterion
    diff_update_w = True
    sim_loss_func = ab_loss
    add_kwargs = {'multilabel':multilabel}
    if loss_type=='minus' or loss_type=='minussample':
        diff_loss_func = minus_loss
    elif loss_type=='minusdiff':
        diff_loss_func = minus_loss
        diff_update_w = False
    elif loss_type=='relative':
        diff_loss_func = relative_loss_mean
        sim_loss_func = rel_loss_mean
    elif loss_type=='relativesample':
        diff_loss_func = relative_loss
        sim_loss_func = rel_loss
        add_kwargs['add_number'] = 1e-6
    elif loss_type=='relmixsample':
        diff_loss_func = relative_loss_s
        sim_loss_func = rel_loss_s
        add_kwargs['add_number'] = 1e-6
    elif loss_type=='relativediff':
        diff_loss_func = relative_loss_mean
        diff_update_w = False
    elif loss_type=='adv':
        diff_loss_func = adv_loss
    elif loss_type=='embed':
        diff_loss_func = adv_loss
        diff_update_w = False
    else:
        print('Unknown loss type for policy training')
        print(loss_type)
        print(adv_criterion)
        raise
    return diff_loss_func,sim_loss_func,diff_update_w,add_kwargs
#noaug regular select
def noaug_select(noaug_reg,extra_criterions,noaug_lossw):
    use_noaug_reg = False
    noaug_target = ''
    if noaug_reg in ['creg','cwreg','cpwreg']:
        print('Using NOAUG regularation ',noaug_reg)
        use_noaug_reg = True
        noaug_lossw = torch.from_numpy(extra_criterions[0].classweight_dist).cuda()
        if extra_criterions[0].reverse_w: #similar to 1/w
            noaug_lossw = noaug_lossw.max() / noaug_lossw
        print('NOAUG regularation class weights',noaug_lossw)
    elif noaug_reg:
        print('Using NOAUG regularation ',noaug_reg)
        print(noaug_lossw)
        use_noaug_reg = True
    
    if noaug_reg=='creg' or noaug_reg=='reg':
        noaug_target = 'p'
    elif noaug_reg=='cwreg':
        noaug_target = 'w'
    elif noaug_reg=='cpwreg':
        noaug_target = 'pw'
    elif noaug_reg=='cdummy':
        noaug_target = 'p'
    return use_noaug_reg, noaug_target, noaug_lossw
#n_policy select
def select_npolicy(policy_list,policy_dist='pwk'):
    #assume [mags:w,weights:p,(keeplen):k,(thres)]
    sel_policy_list = []
    if 'p' in policy_dist:
        sel_policy_list.append(policy_list[1])
    if 'w' in policy_dist:
        sel_policy_list.append(policy_list[0])
    if 'k' in policy_dist and len(policy_list)>2:
        sel_policy_list.append(policy_list[2])
    #print('sel policy shape: ',[n.shape for n in sel_policy_list])
    return torch.cat(sel_policy_list,dim=1)
#difficult search
def difficult_search(input_trsearch,seq_len,target_trsearch,gf_model,adaaug,multilabel,n_class,class_adaptive,search_round,
        diff_mix_feature,diff_update_w,mix_func,adv_criterion,diff_loss_func,lambda_aug,add_kwargs,reweight,
        aug_diff_loss,ori_diff_loss,re_weights_sum,difficult_loss):
    input_trsearch = input_trsearch.float().cuda()
    target_trsearch = target_trsearch.cuda()
    batch_size = target_trsearch.shape[0]
    origin_embed = gf_model.extract_features(input_trsearch, seq_len, pool=False)
    origin_features = gf_model.pool_features(origin_embed)
    origin_logits = gf_model.classify(origin_features)
    policy_y = None
    if class_adaptive: #target to onehot
        if not multilabel:
            policy_y = nn.functional.one_hot(target_trsearch, num_classes=n_class).cuda().float()
        else:
            policy_y = target_trsearch.cuda().float()
    mixed_features, aug_weights = adaaug(input_trsearch, seq_len, mode='explore',mix_feature=diff_mix_feature,y=policy_y,update_w=diff_update_w)
    aug_loss, aug_logits, each_aug_loss = mix_func(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel)
    if len(aug_logits.shape)==4:
        aug_logits = aug_logits.mean(dim=(1,2))
    elif len(aug_logits.shape)==3:
        aug_logits = aug_logits.mean(dim=1)
    ori_loss = cuc_loss(origin_logits,target_trsearch,adv_criterion,multilabel).mean().detach() #!to assert loss mean reduce
    aug_diff_loss += aug_loss.detach().mean().item()
    ori_diff_loss += ori_loss.detach().mean().item()
    loss_prepolicy = diff_loss_func(ori_loss=ori_loss,aug_loss=aug_loss,lambda_aug=lambda_aug,**add_kwargs)
    #print('loss_prepolicy',loss_prepolicy.shape,loss_prepolicy) #!tmp
    if torch.is_tensor(class_weight): #class_weight: (n_class), w_sample: (bs)
        class_weight = class_weight.cuda()
        if not multilabel:
            w_sample = class_weight[target_trsearch].detach()
        else: #class_weight:(1,n_class), target_search:(bs,n_class)=>(bs,n_class) !!!may have some bug!!!
            w_sample = (class_weight.view(1,n_class) * target_trsearch).sum(1)
        tmp = w_sample * loss_prepolicy
        loss_policy = tmp.mean() 
    elif reweight: #reweight part, a,b = ?
        p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].detach()
        p_aug = aug_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].clone().detach()
        w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
        re_weights_sum += w_aug.sum().detach().item()
        if w_aug.sum() > 0:
            w_aug /= (w_aug.mean().detach() + 1e-6)
        else:
            w_aug = 1
        loss_policy = (w_aug * loss_prepolicy).mean() #mean to assert
        #print('w_aug',w_aug)
    else:
        loss_policy = loss_prepolicy.mean() #mean to assert
    #!!!10/13 bug fix!!! ,tmp*4 for same plr
    loss_policy = loss_policy * 4 / search_round
    if torch.any(torch.isnan(loss_policy)) or torch.any(torch.isnan(loss_policy)): #!tmp
        print('loss_policy error: ',loss_policy.detach().item())
        print('ori_loss',ori_loss)
        print('aug_loss',aug_loss)
        print('target_trsearch',target_trsearch)
        print('origin_logits',origin_logits)
        print('aug_logits',aug_logits)
    loss_policy.backward()
    #h_optimizer.step() wait till validation set
    difficult_loss += loss_policy.detach().item()
    torch.cuda.empty_cache()
    return aug_diff_loss,ori_diff_loss,re_weights_sum,difficult_loss

def search_train(args, train_queue, search_queue, tr_search_queue, gf_model, adaaug, criterion, gf_optimizer,scheduler,
            grad_clip, h_optimizer, epoch, search_freq,search_round=1,search_repeat=1, multilabel=False,n_class=10,
            difficult_aug=False,same_train=False,reweight=True,sim_reweight=False,mix_type='embed', warmup_epoch = 0
            ,lambda_sim = 1.0,lambda_aug = 1.0,loss_type='minus',lambda_noaug = 0,train_perfrom = 0.0,noaug_reg='',
            class_adaptive=False,adv_criterion=None,sim_criterion=None,class_weight=None,extra_criterions=[],
            policy_dist='pwk',optim_type='',policy_model=None,
            teacher_model=None,map_select=False,mixup=False,mixup_alpha=1.0,aug_mix=False,visualize=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    confusion_matrix = torch.zeros(n_class,n_class)
    tr_output_matrix = torch.zeros(n_class,n_class).float()
    sea_output_matrix = torch.zeros(n_class,n_class).float()
    tr_embed_matrix = torch.zeros(n_class,gf_model.z_dim).float()
    sea_embed_matrix = torch.zeros(n_class,gf_model.z_dim).float()
    tr_embed_count = torch.zeros(n_class,1)
    sea_embed_count = torch.zeros(n_class,1)
    softmax_m = nn.Softmax(dim=1)
    #tmp for visualize
    n_ops = adaaug.n_ops
    aug_output_matrix = torch.zeros(n_class,n_ops,n_class).float()
    aug_loss_mat = torch.zeros(n_ops).float()
    aug_class_loss = torch.zeros(n_class,n_ops).float()
    #policy_dist = 'pwk' # w=magnitude, p=weight, k=keeplen (if have)
    if policy_dist=='pwk':
        n_policy = adaaug.h_model.proj_out
    else: #single target except threshold
        n_policy = n_ops
    tr_policy_matrix = torch.zeros(n_policy).float()
    sea_policy_matrix = torch.zeros(n_class,n_policy).float()
    
    print(loss_type)
    print(adv_criterion)
    print(f'Lambda Aug {lambda_aug}, Similar {lambda_sim}, NoAug {lambda_noaug}')
    print('Class weight: ',class_weight)
    if adv_criterion==None:
        adv_criterion = criterion
    if sim_criterion==None:
        sim_criterion = criterion
    #noaug criterion selection
    noaug_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    noaug_criterion_w = nn.MSELoss(reduction='none').cuda()
    noaug_lossw = torch.ones(n_class).cuda() * (1.0 - train_perfrom) #first reg
    use_noaug_reg, noaug_target, noaug_lossw = noaug_select(noaug_reg,extra_criterions,noaug_lossw)
    #diff loss criterion
    diff_loss_func, sim_loss_func, diff_update_w, add_kwargs = loss_select(loss_type,adv_criterion,multilabel)
    mix_feature = False
    if mix_type=='embed':
        mix_feature = True
        mix_func = embed_mix
        print('Embed mix type')
    elif mix_type=='output':
        mix_func = output_mix
        print('Output mix type')
    elif mix_type=='loss':
        mix_func = loss_mix
        print('Loss mix type')
    else:
        print('Unknown mix type')
        raise
    if loss_type=='embed':
        diff_mix_feature = False
        mix_func = embed_diff
    else:
        diff_mix_feature = mix_feature
    preds = []
    targets = []
    total = 0
    ex_losses = {str(x.__class__.__name__):0 for x in extra_criterions}
    difficult_loss, adaptive_loss, search_total,re_weights_sum = 0, 0, 0, 0
    aug_diff_loss, ori_diff_loss, aug_search_loss, ori_search_loss = 0,0,0,0
    noaug_reg_sum = 0
    policy_apply = (warmup_epoch==0) #apply policy or not
    for step, (input, seq_len, target) in enumerate(train_queue):
        input = input.float().cuda() #(batch,sed_len,channel)
        target = target.cuda()
        policy_y = None
        if class_adaptive: #target to onehot
            if not multilabel:
                policy_y = nn.functional.one_hot(target, num_classes=n_class).cuda().float()
            else:
                policy_y = target.cuda().float()
        # exploitation
        timer = time.time()
        if epoch>=warmup_epoch:
            aug_images, tr_policy = adaaug(input, seq_len, mode='exploit', y=policy_y, policy_apply=policy_apply)
            #sum of policy, not class wise
            #tr_policy_matrix += torch.cat(tr_policy,dim=1).detach().cpu().sum(0) #[mags,weights], correct
            tr_policy_matrix += select_npolicy(tr_policy,policy_dist=policy_dist).detach().cpu().sum(0) #(n_policy)
        else:
            aug_images = input
        aug_images = aug_images.cuda()
        #mixup if need
        if mixup:
            aug_images, target_a, target_b, mixup_lam = mixup_data(aug_images,target,mixup_alpha)
        if aug_mix:
            aug_images = mixup_aug(aug_images,input,mixup_alpha)
        #start train
        gf_model.train()
        gf_optimizer.zero_grad()
        #logits = gf_model(aug_images, seq_len)
        train_embed = gf_model.extract_features(aug_images, seq_len, pool=True)
        logits = gf_model.classify(train_embed)
        #if mixup
        if mixup:
            aug_loss = mixup_lam * cuc_loss(logits,target_a,criterion,multilabel) + (1.0-mixup_lam) * cuc_loss(logits,target_b,criterion,multilabel)
        else:
            aug_loss = cuc_loss(logits,target,criterion,multilabel)
        #difficult aug
        ori_loss = 0
        batch_size = target.shape[0]
        if not same_train and difficult_aug:
            origin_logits = gf_model(input, seq_len)
            if multilabel:
                ori_loss = criterion(origin_logits, target.float()) / 2
            else:
                ori_loss = criterion(origin_logits, target.long()) / 2
            if reweight: #reweight part, a,b = ?
                p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target].detach()
                p_aug = logits.softmax(dim=1)[torch.arange(batch_size), target].clone().detach()
                w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
                if w_aug.sum() > 0:
                    w_aug /= (w_aug.mean().detach() + 1e-6)
                else:
                    w_aug = 1
                aug_loss = (w_aug * lambda_aug * aug_loss).mean() / 2
            else:
                aug_loss = (lambda_aug * aug_loss).mean() / 2
        loss = ori_loss + aug_loss.mean() #assert
        loss.backward()
        nn.utils.clip_grad_norm_(gf_model.parameters(), grad_clip)
        h_optimizer.zero_grad() #12/18 add, assert h_model weight not update
        gf_optimizer.step()
        scheduler.step() #8/03 add
        gf_optimizer.zero_grad()
        torch.cuda.empty_cache()
        #ema teacher
        if teacher_model:
            teacher_model.update_parameters(gf_model)
        #  stats
        n = target.size(0)
        objs.update(loss.detach().item(), n)
        if not multilabel:
            prec1, prec5 = utils.accuracy(logits.detach(), target.detach(), topk=(1, 5))
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)
            _, predicted = torch.max(logits.data, 1)
            soft_out = softmax_m(logits).detach().cpu()
            embed_out = train_embed.detach().cpu()
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            for i,t in enumerate(target.data.view(-1)):
                tr_output_matrix[t.long(),:] += soft_out[i]
                tr_embed_matrix[t.long(),:] += embed_out[i] #11/14 add
                tr_embed_count[t.long(),0] += 1
        else:
            predicted = torch.sigmoid(logits)
        
        #ACC-AUROC
        preds.append(predicted.cpu().detach())
        targets.append(target.cpu().detach())
        exploitation_time = time.time() - timer

        # exploration
        timer = time.time()
        if epoch>= warmup_epoch and step % search_freq == search_freq-1: #warmup
            policy_apply = True #start learning policy
            #difficult, train input, target
            gf_model.eval() #11/9 add
            gf_optimizer.zero_grad()
            h_optimizer.zero_grad()
            input_search_list,seq_len_list,target_search_list,policy_y_list = [],[],[],[]
            for r in range(search_round): #grad accumulation
                if difficult_aug:
                    if r % search_repeat==0:
                        input_trsearch, seq_len, target_trsearch = next(iter(tr_search_queue))
                    input_trsearch = input_trsearch.float().cuda()
                    target_trsearch = target_trsearch.cuda()
                    batch_size = target_trsearch.shape[0]
                    origin_embed = gf_model.extract_features(input_trsearch, seq_len, pool=False)
                    origin_features = gf_model.pool_features(origin_embed)
                    origin_logits = gf_model.classify(origin_features)
                    policy_y = None
                    if class_adaptive: #target to onehot
                        if not multilabel:
                            policy_y = nn.functional.one_hot(target_trsearch, num_classes=n_class).cuda().float()
                        else:
                            policy_y = target_trsearch.cuda().float()
                    mixed_features, aug_weights = adaaug(input_trsearch, seq_len, mode='explore',mix_feature=diff_mix_feature,y=policy_y,update_w=diff_update_w)
                    aug_loss, aug_logits, each_aug_loss = mix_func(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel)
                    if len(aug_logits.shape)==4:
                        aug_logits = aug_logits.mean(dim=(1,2))
                    elif len(aug_logits.shape)==3:
                        aug_logits = aug_logits.mean(dim=1)
                    ori_loss = cuc_loss(origin_logits,target_trsearch,adv_criterion,multilabel).mean().detach() #!to assert loss mean reduce
                    aug_diff_loss += aug_loss.detach().mean().item()
                    ori_diff_loss += ori_loss.detach().mean().item()
                    loss_prepolicy = diff_loss_func(ori_loss=ori_loss,aug_loss=aug_loss,lambda_aug=lambda_aug,**add_kwargs)
                    #print('loss_prepolicy',loss_prepolicy.shape,loss_prepolicy) #!tmp
                    if torch.is_tensor(class_weight): #class_weight: (n_class), w_sample: (bs)
                        class_weight = class_weight.cuda()
                        if not multilabel:
                            w_sample = class_weight[target_trsearch].detach()
                        else: #class_weight:(1,n_class), target_search:(bs,n_class)=>(bs,n_class) !!!may have some bug!!!
                            w_sample = (class_weight.view(1,n_class) * target_trsearch).sum(1)
                        tmp = w_sample * loss_prepolicy
                        loss_policy = tmp.mean() 
                    elif reweight: #reweight part, a,b = ?
                        p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].detach()
                        p_aug = aug_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].clone().detach()
                        w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
                        re_weights_sum += w_aug.sum().detach().item()
                        if w_aug.sum() > 0:
                            w_aug /= (w_aug.mean().detach() + 1e-6)
                        else:
                            w_aug = 1
                        loss_policy = (w_aug * loss_prepolicy).mean() #mean to assert
                        #print('w_aug',w_aug)
                    else:
                        loss_policy = loss_prepolicy.mean() #mean to assert
                    #!!!10/13 bug fix!!! ,tmp*4 for same plr
                    loss_policy = loss_policy * 4 / search_round
                    if torch.any(torch.isnan(loss_policy)) or torch.any(torch.isnan(loss_policy)): #!tmp
                        print('loss_policy error: ',loss_policy.detach().item())
                        print('ori_loss',ori_loss)
                        print('aug_loss',aug_loss)
                        print('target_trsearch',target_trsearch)
                        print('origin_logits',origin_logits)
                        print('aug_logits',aug_logits)

                    loss_policy.backward()
                    #h_optimizer.step() wait till validation set
                    difficult_loss += loss_policy.detach().item()
                    torch.cuda.empty_cache()
                #similar
                if r % search_repeat==0:
                    input_search, seq_len, target_search = next(iter(search_queue))
                input_search = input_search.float().cuda()
                target_search = target_search.cuda()
                search_bs = target_search.shape[0]
                policy_y = None
                if class_adaptive: #target to onehot
                    if not multilabel:
                        policy_y = nn.functional.one_hot(target_search, num_classes=n_class).cuda().float()
                    else:
                        policy_y = target_search.cuda().float()
                policy_y_list.append(policy_y)
                mixed_features, aug_weights = adaaug(input_search, seq_len, mode='explore',mix_feature=mix_feature,y=policy_y)
                #weight regular to NOAUG, [weight,mag] different from other policy output
                aug_weight = aug_weights[0] # (bs, n_ops), 0 is NOAUG
                aug_magnitude = aug_weights[1]
                #aug_policy = torch.cat([aug_weights[1],aug_weights[0]],dim=1) #h_model output is mag,weight
                aug_policy = select_npolicy([aug_weights[1],aug_weights[0]],policy_dist=policy_dist)
                noaug_mag_target = torch.zeros(aug_magnitude.shape).cuda().float()
                if noaug_reg=='cdummy':
                    print('Use taget to fake noaug target')
                    noaug_w_target = target_search.detach().long()
                else:
                    noaug_w_target = torch.zeros(aug_weight.shape[0]).cuda().long()
                if use_noaug_reg: #need test
                    #noaug_loss = lambda_noaug * (1.0 - train_perfrom) * noaug_criterion(aug_weight,noaug_target) #10/26 change
                    noaug_samplew = noaug_lossw[target_search.detach()]
                    noaug_loss = 0
                    if 'w' in noaug_target:
                        tmp_loss = (lambda_noaug * noaug_samplew * noaug_criterion_w(aug_magnitude,noaug_mag_target).mean(1)).mean() #mean to assert loss is scalar
                        noaug_loss += tmp_loss
                        print('magnitude loss: ',tmp_loss.detach().item())
                    if 'p' in noaug_target:
                        tmp_loss = (lambda_noaug * noaug_samplew * noaug_criterion(aug_weight,noaug_w_target)).mean()
                        noaug_loss += tmp_loss
                        print('prob loss: ',tmp_loss.detach().item())
                    if torch.is_tensor(noaug_loss):
                        noaug_reg_sum += noaug_loss.detach().mean().item()
                    #print('noaug regular sum: ',noaug_reg_sum)
                else:
                    noaug_loss = 0
                #tea
                if teacher_model==None:
                    sim_model = gf_model
                else:
                    sim_model = teacher_model.module
                #sim mix if need, calculate loss
                augsear_loss, logits_search, each_aug_loss = mix_func(gf_model,mixed_features,aug_weights,sim_criterion,target_search,multilabel)
                origin_embed = sim_model.extract_features(input_search, seq_len, pool=True)
                origin_logits = sim_model.classify(origin_embed)
                ori_loss = cuc_loss(origin_logits,target_search,sim_criterion,multilabel)
                #ori_loss = ori_loss.mean() #!to assert loss mean reduce
                #output pred
                if not multilabel:
                    if mix_type=='loss': #tmp for visualize
                        if len(logits_search.shape)==4:
                            logits_search_avg = logits_search.mean(dim=(1,2))
                        else:
                            logits_search_avg = logits_search.mean(dim=1)
                        soft_out = softmax_m(logits_search_avg).detach().cpu() #(bs,n_class) for embed or output mix
                        soft_all = softmax_m(logits_search.reshape(-1,n_class)).detach().cpu().reshape(*logits_search.shape) #(bs,keep_len,n_ops)
                        origin_embed_out = origin_embed.detach().cpu()
                        each_aug_loss = each_aug_loss.detach().cpu()
                        #print('soft all ',soft_all.shape)
                        #print('each_aug_loss ',each_aug_loss.shape)
                        for i,t in enumerate(target_search.data.view(-1)):
                            sea_output_matrix[t.long()] += soft_out[i]
                            sea_embed_matrix[t.long(),:] += origin_embed_out[i] #!!! 11/14 add, use origin
                            sea_embed_count[t.long(),0] += 1
                            for aug_idx in range(n_ops):
                                aug_output_matrix[t.long(),aug_idx] += soft_all[i,aug_idx]
                                aug_loss_mat[aug_idx] += each_aug_loss[i,aug_idx]
                                aug_class_loss[t.long(),aug_idx] += each_aug_loss[i,aug_idx]
                    else:
                            soft_out = softmax_m(logits_search).detach().cpu() #(bs,n_class) for embed or output mix
                            origin_embed_out = origin_embed.detach().cpu()
                            for i,t in enumerate(target_search.data.view(-1)):
                                sea_output_matrix[t.long()] += soft_out[i]
                                sea_embed_matrix[t.long(),:] += origin_embed_out[i] #!!! 11/14 add, use origin
                                sea_embed_count[t.long(),0] += 1

                #similar reweight?
                aug_search_loss += augsear_loss.detach().mean().item()
                ori_search_loss += ori_loss.detach().mean().item()
                loss = sim_loss_func(ori_loss,augsear_loss,**add_kwargs)
                #print(loss.shape,loss) #!tmp
                if torch.is_tensor(class_weight): #class_weight: (n_class), w_sample: (bs)
                    class_weight = class_weight.cuda()
                    if not multilabel:
                        w_sample = class_weight[target_search].detach()
                    else: #class_weight:(1,n_class), target_search:(bs,n_class)
                        w_sample = (class_weight.view(1,n_class) * target_search).sum(1)
                    print('w_sample: ',w_sample.shape)
                    tmp = w_sample * loss
                    print('sim loss: ',tmp)
                    loss = tmp.mean() 
                elif sim_reweight: #reweight part, a,b = ?
                    p_orig = origin_logits.softmax(dim=1)[torch.arange(search_bs), target_search].detach()
                    p_aug = logits_search.softmax(dim=1)[torch.arange(search_bs), target_search].clone().detach()
                    w_aug = torch.sqrt((1.0 - p_orig)) #a=0.5,b=0.5
                    if w_aug.sum() > 0:
                        w_aug /= (w_aug.mean().detach() + 1e-6)
                    else:
                        w_aug = 1
                    loss = (w_aug * loss).mean()
                    #print('Similar loss:', loss.detach().item())
                else:
                    loss = loss.mean()
                    #print('Similar loss:', loss.detach().item())
                loss = loss * lambda_sim + noaug_loss
                #extra losses
                for e_criterion in extra_criterions:
                    if e_criterion.loss_target=='output': #for class distance loss target
                        e_loss = e_criterion(logits_search,target_search,sim_targets=origin_logits)
                    elif e_criterion.loss_target=='embed': 
                        e_loss = e_criterion(mixed_features,target_search,sim_targets=origin_embed) #using mix_fearures as loss
                    elif e_criterion.loss_target=='policy': 
                        e_loss = e_criterion(aug_policy,target_search,sim_targets=origin_embed) #using mix_fearures as loss
                    else:
                        print('Unknown loss target: ',e_criterion.loss_target)
                        raise
                    if torch.is_tensor(e_loss):
                        loss += e_loss.mean()
                        ex_losses[str(e_criterion.__class__.__name__)] += e_loss.detach().item()
                        print('Extra class distance loss:', e_loss.detach().item())
                #!!!10/13 bug fix, tmp *4 for plr!!!
                loss = loss * 4 / search_round
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)): #!tmp
                    print('Loss error: ',loss.detach().item())
                    print('ori_loss',ori_loss)
                    print('augsear_loss',augsear_loss)
                    print('target_search',target_search)
                    print('origin_logits',origin_logits)
                    print('logits_search',logits_search)
                    print('noaug_loss',noaug_loss)

                loss.backward()
                adaptive_loss += loss.detach().item()
                search_total += 1
                input_search_list.append(input_search.detach())
                seq_len_list.append(seq_len.detach())
                target_search_list.append(target_search.detach())
                #torch.cuda.empty_cache()
            #accumulation update
            nn.utils.clip_grad_norm_(adaaug.h_model.parameters(), grad_clip)
            h_optimizer.step()
            #  log policy
            gf_optimizer.zero_grad()
            h_optimizer.zero_grad()
            input_search_list = torch.cat(input_search_list,dim=0)
            seq_len_list = torch.cat(seq_len_list,dim=0)
            target_search_list = torch.cat(target_search_list,dim=0)
            if class_adaptive:
                policy_y_list = torch.cat(policy_y_list,dim=0)
            else:
                policy_y_list = None
            policy = adaaug.add_history(input_search_list, seq_len_list, target_search_list,y=policy_y_list)
            #back to policy
            #policy_out = torch.cat(policy,dim=1).detach().cpu() #[mag, weight] is same with h_model output
            policy_out = select_npolicy(policy,policy_dist=policy_dist).detach().cpu()
            sea_policy_matrix += policy_out #sum of policy for each class

        exploration_time = time.time() - timer
        torch.cuda.empty_cache()
        #log
        global_step = epoch * len(train_queue) + step
        if global_step % args.report_freq == 0:
            logging.info('  |train %03d %e %f %f | %.3f + %.3f s', global_step,
                objs.avg, top1.avg, top5.avg, exploitation_time, exploration_time)
    #visualize
    if visualize:
        input_search, seq_len, target_search = next(iter(search_queue))
        input_search = input_search.float().cuda()
        target_search = target_search.cuda()
        policy_y = None
        if class_adaptive: #target to onehot
            if not multilabel:
                policy_y = nn.functional.one_hot(target_search, num_classes=n_class).cuda().float()
            else:
                policy_y = target_search.cuda().float()
        adaaug.visualize_result(input_search, seq_len,policy_y,target_search)
        exit()
    #class-wise & Total
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/torch.clamp(confusion_matrix.sum(1),min=1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / torch.clamp(confusion_matrix.sum(),min=1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
        ptype = 'acc'
        logging.info(f'Epoch train: loss={objs.avg} top1acc={top1.avg} top5acc={top5.avg}')
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom_cw2 = utils.mAP_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom_cw2.mean()
        logging.info('Epoch train: loss=%e macroAUROC=%f', objs.avg, perfrom)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Epoch train: loss=%e macroAP=%f', objs.avg, perfrom2)
        logging.info('class-wise AP: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw2])+']')
        ptype = 'auroc'
    #table dic
    table_dic = {}
    table_dic['search_output'] = (sea_output_matrix / torch.clamp(sea_output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic['train_output'] = (tr_output_matrix / torch.clamp(tr_output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic['train_embed'] = tr_embed_matrix / torch.clamp(tr_embed_count.float(),min=1e-9)
    table_dic['train_policy'] = tr_policy_matrix / tr_embed_count.sum(0)
    table_dic['search_embed'] = sea_embed_matrix / torch.clamp(sea_embed_count.float(),min=1e-9)
    table_dic['search_policy'] = sea_policy_matrix / torch.clamp(sea_embed_count.float(),min=1e-9)
    #sample
    print('sample policy class 0: ',table_dic['search_policy'][0])
    #average policy for all train+search data
    table_dic['train_confusion'] = confusion_matrix
    #tmp for aug loss visual
    if mix_type=='loss':
        table_dic['aug_output'] = aug_output_matrix / torch.clamp(aug_output_matrix.sum(dim=-1,keepdim=True),min=1e-9)
        table_dic['aug_loss'] = aug_loss_mat / torch.sum(sea_embed_count.float())
        table_dic['class_aug_loss'] = aug_class_loss / torch.clamp(sea_embed_count.float(),min=1e-9)
        print('aug_output: ',table_dic['aug_output'].shape) #tmp
        print('aug_loss: ',table_dic['aug_loss'].shape) #tmp
        print('class_aug_loss: ',table_dic['class_aug_loss'].shape) #tmp
        print('aug_output: ',table_dic['aug_output']) #tmp
        print('aug_loss: ',table_dic['aug_loss']) #tmp
        print('class_aug_loss: ',table_dic['class_aug_loss']) #tmp
    #wandb dic
    out_dic = {}
    out_dic['train_loss'] = objs.avg
    for k in sorted(ex_losses):
        out_dic[k] = ex_losses[k] / search_total
    if epoch>= warmup_epoch:
        out_dic['adaptive_loss'] = adaptive_loss * search_round / (search_total*4)
        out_dic['aug_sear_loss'] = aug_search_loss / search_total
        out_dic['ori_sear_loss'] = ori_search_loss / search_total
        if difficult_aug:
            out_dic['difficult_loss'] = difficult_loss * search_round / (search_total*4)
            out_dic['reweight_sum'] = re_weights_sum / search_total
            out_dic['aug_diff_loss'] = aug_diff_loss / search_total
            out_dic['ori_diff_loss'] = ori_diff_loss / search_total
            out_dic['search_loss'] = out_dic['adaptive_loss']+out_dic['difficult_loss']
        else:
            out_dic['search_loss'] = out_dic['adaptive_loss']
        if use_noaug_reg:
            out_dic['noaug_reg'] = noaug_reg_sum / search_total
        
    out_dic[f'train_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'train_{ptype}_c{i}'] = e_c
    if multilabel: #addition
        add_type = 'ap'
        out_dic[f'train_{add_type}_avg'] = perfrom2
        for i,e_c in enumerate(perfrom_cw2):
            out_dic[f'train_{add_type}_c{i}'] = e_c
        if map_select:
            return perfrom2, objs.avg, out_dic, table_dic

    return perfrom, objs.avg, out_dic, table_dic

def search_infer(valid_queue, gf_model, criterion, multilabel=False, n_class=10,mode='test',map_select=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    gf_model.eval()
    confusion_matrix = torch.zeros(n_class,n_class)
    output_matrix = torch.zeros(n_class,n_class).float()
    embed_matrix = torch.zeros(n_class,256).float() #tmp !!!
    embed_count = torch.zeros(n_class,1)
    softmax_m = nn.Softmax(dim=1)
    preds = []
    targets = []
    preds_score = []
    targets_score = []
    total = 0
    with torch.no_grad():
        for input, seq_len, target in valid_queue:
            input = input.float().cuda()
            target = target.cuda()
            embed = gf_model.extract_features(input, seq_len, pool=True)
            logits = gf_model.classify(embed)
            #logits = gf_model(input,seq_len)
            if multilabel:
                loss = criterion(logits, target.float())
            else:
                loss = criterion(logits, target.long())

            n = input.size(0)
            objs.update(loss.detach().item(), n)
            if not multilabel:
                _, predicted = torch.max(logits.data, 1)
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                top1.update(prec1.detach().item(), n)
                top5.update(prec5.detach().item(), n)
                soft_out = softmax_m(logits).detach().cpu()
                embed_out = embed.detach().cpu()
                for t, p in zip(target.data.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                for i,t in enumerate(target.data.view(-1)):
                    output_matrix[t.long(),:] += soft_out[i]
                    embed_matrix[t.long(),:] += embed_out[i] #11/14 add
                    embed_count[t.long(),0] += 1
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                each_pscore = torch.gather(soft_out,1,predicted.cpu().detach().long().view(-1,1))
                each_tscore = torch.gather(soft_out,1,target.cpu().detach().long().view(-1,1))
                preds_score.append(each_pscore) #(bs,n_class)[(bs)]=>(bs)
                targets_score.append(each_tscore)
            else:
                predicted = torch.sigmoid(logits.data)
                preds_score.append(predicted) #multilabel don't need
                targets_score.append(predicted) #multilabel don't need
            preds.append(predicted.cpu().detach())
            targets.append(target.cpu().detach())
    #table dic
    table_dic = {}
    table_dic[f'{mode}_output'] = (output_matrix / torch.clamp(output_matrix.sum(dim=1,keepdim=True),1e-9))
    table_dic[f'{mode}_embed'] = embed_matrix / torch.clamp(embed_count.float(),min=1e-9)
    table_dic[f'{mode}_confusion'] = confusion_matrix
    #class-wise
    targets_np = torch.cat(targets).numpy()
    preds_np = torch.cat(preds).numpy()
    targets_score_np = torch.cat(targets_score).numpy()
    preds_score_np = torch.cat(preds_score).numpy()
    table_dic[f'{mode}_target'] = targets_np
    table_dic[f'{mode}_predict'] = preds_np
    table_dic[f'{mode}_target_score'] = targets_score_np
    table_dic[f'{mode}_predict_score'] = preds_score_np
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/torch.clamp(confusion_matrix.sum(1),min=1e-9)
        logging.info(f'{mode} class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / torch.clamp(confusion_matrix.sum(),min=1e-9)
        logging.info('%s Overall Acc: %f',mode,nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
        ptype = 'acc'
    else:
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom_cw2 = utils.mAP_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom_cw2.mean()
        logging.info('Epoch %s: loss=%e macroAUROC=%f', mode, objs.avg, perfrom)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Epoch %s: loss=%e macroAP=%f',mode, objs.avg, perfrom2)
        logging.info('class-wise AP: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw2])+']')
        ptype = 'auroc'
    
    #wandb dic
    out_dic = {}
    out_dic[f'{mode}_loss'] = objs.avg
    out_dic[f'{mode}_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'{mode}_{ptype}_c{i}'] = e_c
    if multilabel: #addition
        add_type = 'ap'
        out_dic[f'{mode}_{add_type}_avg'] = perfrom2
        for i,e_c in enumerate(perfrom_cw2):
            out_dic[f'{mode}_{add_type}_c{i}'] = e_c
        if map_select:
            return perfrom2, objs.avg, out_dic, table_dic
    
    return perfrom, objs.avg, out_dic, table_dic
    #return top1.avg, objs.avg

### only for tmp use on new grad method, 12/14
def search_train_neumann(args, train_queue, search_queue, tr_search_queue, gf_model, adaaug, criterion, gf_optimizer,scheduler,
            grad_clip, h_optimizer, epoch, search_freq,search_round=1,search_repeat=1, multilabel=False,n_class=10,
            difficult_aug=False,same_train=False,reweight=True,sim_reweight=False,mix_type='embed', warmup_epoch = 0
            ,lambda_sim = 1.0,lambda_aug = 1.0,loss_type='minus',lambda_noaug = 0,train_perfrom = 0.0,noaug_reg='',
            class_adaptive=False,adv_criterion=None,sim_criterion=None,class_weight=None,extra_criterions=[],
            policy_dist='pwk',optim_type='',
            teacher_model=None,map_select=False,mixup=False,mixup_alpha=1.0,aug_mix=False,visualize=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    confusion_matrix = torch.zeros(n_class,n_class)
    tr_output_matrix = torch.zeros(n_class,n_class).float()
    sea_output_matrix = torch.zeros(n_class,n_class).float()
    tr_embed_matrix = torch.zeros(n_class,gf_model.z_dim).float()
    sea_embed_matrix = torch.zeros(n_class,gf_model.z_dim).float()
    tr_embed_count = torch.zeros(n_class,1)
    sea_embed_count = torch.zeros(n_class,1)
    softmax_m = nn.Softmax(dim=1)
    #tmp for visualize
    n_ops = adaaug.n_ops
    aug_output_matrix = torch.zeros(n_class,n_ops,n_class).float()
    aug_loss_mat = torch.zeros(n_ops).float()
    aug_class_loss = torch.zeros(n_class,n_ops).float()
    #policy_dist = 'pwk' # w=magnitude, p=weight, k=keeplen (if have)
    if policy_dist=='pwk':
        n_policy = adaaug.h_model.proj_out
    else: #single target except threshold
        n_policy = n_ops
    tr_policy_matrix = torch.zeros(n_policy).float()
    sea_policy_matrix = torch.zeros(n_class,n_policy).float()
    
    print(loss_type)
    print(adv_criterion)
    print(f'Neumann gradient matching method, only for debug and test')
    print(f'Lambda Aug {lambda_aug}, Similar {lambda_sim}, NoAug {lambda_noaug}')
    print('Class weight: ',class_weight)
    if adv_criterion==None:
        adv_criterion = criterion
    if sim_criterion==None:
        sim_criterion = criterion
    #noaug criterion selection
    noaug_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    noaug_criterion_w = nn.MSELoss(reduction='none').cuda()
    noaug_lossw = torch.ones(n_class).cuda() * (1.0 - train_perfrom) #first reg
    use_noaug_reg, noaug_target, noaug_lossw = noaug_select(noaug_reg,extra_criterions,noaug_lossw)
    #diff loss criterion
    diff_loss_func, sim_loss_func, diff_update_w, add_kwargs = loss_select(loss_type,adv_criterion,multilabel)
    mix_feature = False
    if mix_type=='embed':
        mix_feature = True
        mix_func = embed_mix
        print('Embed mix type')
    elif mix_type=='output':
        mix_func = output_mix
        print('Output mix type')
    elif mix_type=='loss':
        mix_func = loss_mix
        print('Loss mix type')
    else:
        print('Unknown mix type')
        raise
    if loss_type=='embed':
        diff_mix_feature = False
        mix_func = embed_diff
    else:
        diff_mix_feature = mix_feature
    preds = []
    targets = []
    total = 0
    ex_losses = {str(x.__class__.__name__):0 for x in extra_criterions}
    difficult_loss, adaptive_loss, search_total,re_weights_sum = 0, 0, 0, 0
    aug_diff_loss, ori_diff_loss, aug_search_loss, ori_search_loss = 0,0,0,0
    hyper_grad_avg = 0
    noaug_reg_sum = 0
    policy_apply = (warmup_epoch==0) #apply policy or not
    for step, (input, seq_len, target) in enumerate(train_queue):
        input = input.float().cuda() #(batch,sed_len,channel)
        target = target.cuda()
        policy_y = None
        if class_adaptive: #target to onehot
            if not multilabel:
                policy_y = nn.functional.one_hot(target, num_classes=n_class).cuda().float()
            else:
                policy_y = target.cuda().float()
        # exploitation
        timer = time.time()
        if epoch>=warmup_epoch:
            aug_images, tr_policy = adaaug(input, seq_len, mode='exploit', y=policy_y, policy_apply=policy_apply)
            #sum of policy, not class wise
            #tr_policy_matrix += torch.cat(tr_policy,dim=1).detach().cpu().sum(0) #[mags,weights], correct
            tr_policy_matrix += select_npolicy(tr_policy,policy_dist=policy_dist).detach().cpu().sum(0) #(n_policy)
        else:
            aug_images = input
        aug_images = aug_images.cuda()
        #mixup if need
        if mixup:
            aug_images, target_a, target_b, mixup_lam = mixup_data(aug_images,target,mixup_alpha)
        if aug_mix:
            aug_images = mixup_aug(aug_images,input,mixup_alpha)
        #start train
        gf_model.train()
        gf_optimizer.zero_grad()
        #logits = gf_model(aug_images, seq_len)
        train_embed = gf_model.extract_features(aug_images, seq_len, pool=True)
        logits = gf_model.classify(train_embed)
        #if mixup
        if mixup:
            aug_loss = mixup_lam * cuc_loss(logits,target_a,criterion,multilabel) + (1.0-mixup_lam) * cuc_loss(logits,target_b,criterion,multilabel)
        else:
            aug_loss = cuc_loss(logits,target,criterion,multilabel)
        #difficult aug
        ori_loss = 0
        batch_size = target.shape[0]
        if not same_train and difficult_aug:
            origin_logits = gf_model(input, seq_len)
            if multilabel:
                ori_loss = criterion(origin_logits, target.float()) / 2
            else:
                ori_loss = criterion(origin_logits, target.long()) / 2
            if reweight: #reweight part, a,b = ?
                p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target].detach()
                p_aug = logits.softmax(dim=1)[torch.arange(batch_size), target].clone().detach()
                w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
                if w_aug.sum() > 0:
                    w_aug /= (w_aug.mean().detach() + 1e-6)
                else:
                    w_aug = 1
                aug_loss = (w_aug * lambda_aug * aug_loss).mean() / 2
            else:
                aug_loss = (lambda_aug * aug_loss).mean() / 2
        loss = ori_loss + aug_loss
        loss.backward()
        nn.utils.clip_grad_norm_(gf_model.parameters(), grad_clip)
        gf_optimizer.step()
        scheduler.step() #8/03 add
        gf_optimizer.zero_grad()
        torch.cuda.empty_cache()
        #ema teacher
        if teacher_model:
            teacher_model.update_parameters(gf_model)
        #  stats
        n = target.size(0)
        objs.update(loss.detach().item(), n)
        if not multilabel:
            prec1, prec5 = utils.accuracy(logits.detach(), target.detach(), topk=(1, 5))
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)
            _, predicted = torch.max(logits.data, 1)
            soft_out = softmax_m(logits).detach().cpu()
            embed_out = train_embed.detach().cpu()
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            for i,t in enumerate(target.data.view(-1)):
                tr_output_matrix[t.long(),:] += soft_out[i]
                tr_embed_matrix[t.long(),:] += embed_out[i] #11/14 add
                tr_embed_count[t.long(),0] += 1
        else:
            predicted = torch.sigmoid(logits)
        
        #ACC-AUROC
        preds.append(predicted.cpu().detach())
        targets.append(target.cpu().detach())
        exploitation_time = time.time() - timer

        # exploration
        timer = time.time()
        if epoch>= warmup_epoch and step % search_freq == search_freq-1: #warmup
            policy_apply = True #start learning policy
            #difficult, train input, target
            gf_model.eval() #11/9 add
            gf_optimizer.zero_grad()
            h_optimizer.zero_grad()
            #input_search_list,seq_len_list,target_search_list,policy_y_list = [],[],[],[]
            #gradient match hyper step
            #current train lr
            for param_group in gf_optimizer.param_groups:
                elementary_lr = param_group['lr']
                break
            neum_steps = 1 #follow taskaug
            use_device = torch.device('cuda')
            hyp_params = list(adaaug.h_model.parameters())
            hyper_grad,diff_loss,sea_loss,input_search_list,seq_len_list,target_search_list,policy_y_list = \
                hyper_step(gf_model,adaaug,hyp_params,tr_search_queue,gf_optimizer,search_queue,elementary_lr,
                neum_steps,use_device,sim_criterion,n_class,search_round,class_adaptive,multilabel,update_w=diff_update_w)
            aug_diff_loss += diff_loss
            aug_search_loss += sea_loss
            hyper_grad_avg += hyper_grad.detach().mean().item()
            search_total += 1
            '''for r in range(search_round): #grad accumulation
                if difficult_aug:
                    if r % search_repeat==0:
                        input_trsearch, seq_len, target_trsearch = next(iter(tr_search_queue))
                    input_trsearch = input_trsearch.float().cuda()
                    target_trsearch = target_trsearch.cuda()
                    batch_size = target_trsearch.shape[0]
                    origin_embed = gf_model.extract_features(input_trsearch, seq_len, pool=False)
                    origin_features = gf_model.pool_features(origin_embed)
                    origin_logits = gf_model.classify(origin_features)
                    policy_y = None
                    if class_adaptive: #target to onehot
                        if not multilabel:
                            policy_y = nn.functional.one_hot(target_trsearch, num_classes=n_class).cuda().float()
                        else:
                            policy_y = target_trsearch.cuda().float()
                    mixed_features, aug_weights = adaaug(input_trsearch, seq_len, mode='explore',mix_feature=diff_mix_feature,y=policy_y,update_w=diff_update_w)
                    aug_loss, aug_logits, each_aug_loss = mix_func(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel)
                    if len(aug_logits.shape)==4:
                        aug_logits = aug_logits.mean(dim=(1,2))
                    elif len(aug_logits.shape)==3:
                        aug_logits = aug_logits.mean(dim=1)
                    ori_loss = cuc_loss(origin_logits,target_trsearch,adv_criterion,multilabel).mean().detach() #!to assert loss mean reduce
                    aug_diff_loss += aug_loss.detach().mean().item()
                    ori_diff_loss += ori_loss.detach().mean().item()
                    loss_prepolicy = diff_loss_func(ori_loss=ori_loss,aug_loss=aug_loss,lambda_aug=lambda_aug,**add_kwargs)
                    #print('loss_prepolicy',loss_prepolicy.shape,loss_prepolicy) #!tmp
                    if torch.is_tensor(class_weight): #class_weight: (n_class), w_sample: (bs)
                        class_weight = class_weight.cuda()
                        if not multilabel:
                            w_sample = class_weight[target_trsearch].detach()
                        else: #class_weight:(1,n_class), target_search:(bs,n_class)=>(bs,n_class) !!!may have some bug!!!
                            w_sample = (class_weight.view(1,n_class) * target_trsearch).sum(1)
                        tmp = w_sample * loss_prepolicy
                        loss_policy = tmp.mean() 
                    elif reweight: #reweight part, a,b = ?
                        p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].detach()
                        p_aug = aug_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].clone().detach()
                        w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
                        re_weights_sum += w_aug.sum().detach().item()
                        if w_aug.sum() > 0:
                            w_aug /= (w_aug.mean().detach() + 1e-6)
                        else:
                            w_aug = 1
                        loss_policy = (w_aug * loss_prepolicy).mean() #mean to assert
                        #print('w_aug',w_aug)
                    else:
                        loss_policy = loss_prepolicy.mean() #mean to assert
                    #!!!10/13 bug fix!!! ,tmp*4 for same plr
                    loss_policy = loss_policy * 4 / search_round
                    if torch.any(torch.isnan(loss_policy)) or torch.any(torch.isnan(loss_policy)): #!tmp
                        print('loss_policy error: ',loss_policy.detach().item())
                        print('ori_loss',ori_loss)
                        print('aug_loss',aug_loss)
                        print('target_trsearch',target_trsearch)
                        print('origin_logits',origin_logits)
                        print('aug_logits',aug_logits)

                    loss_policy.backward()
                    #h_optimizer.step() wait till validation set
                    difficult_loss += loss_policy.detach().item()
                    torch.cuda.empty_cache()
                #similar
                if r % search_repeat==0:
                    input_search, seq_len, target_search = next(iter(search_queue))
                input_search = input_search.float().cuda()
                target_search = target_search.cuda()
                search_bs = target_search.shape[0]
                policy_y = None
                if class_adaptive: #target to onehot
                    if not multilabel:
                        policy_y = nn.functional.one_hot(target_search, num_classes=n_class).cuda().float()
                    else:
                        policy_y = target_search.cuda().float()
                policy_y_list.append(policy_y)
                mixed_features, aug_weights = adaaug(input_search, seq_len, mode='explore',mix_feature=mix_feature,y=policy_y)
                #weight regular to NOAUG, [weight,mag] different from other policy output
                aug_weight = aug_weights[0] # (bs, n_ops), 0 is NOAUG
                aug_magnitude = aug_weights[1]
                #aug_policy = torch.cat([aug_weights[1],aug_weights[0]],dim=1) #h_model output is mag,weight
                aug_policy = select_npolicy([aug_weights[1],aug_weights[0]],policy_dist=policy_dist)
                noaug_mag_target = torch.zeros(aug_magnitude.shape).cuda().float()
                if noaug_reg=='cdummy':
                    print('Use taget to fake noaug target')
                    noaug_w_target = target_search.detach().long()
                else:
                    noaug_w_target = torch.zeros(aug_weight.shape[0]).cuda().long()
                if use_noaug_reg: #need test
                    #noaug_loss = lambda_noaug * (1.0 - train_perfrom) * noaug_criterion(aug_weight,noaug_target) #10/26 change
                    noaug_samplew = noaug_lossw[target_search.detach()]
                    noaug_loss = 0
                    if 'w' in noaug_target:
                        tmp_loss = (lambda_noaug * noaug_samplew * noaug_criterion_w(aug_magnitude,noaug_mag_target).mean(1)).mean() #mean to assert loss is scalar
                        noaug_loss += tmp_loss
                        print('magnitude loss: ',tmp_loss.detach().item())
                    if 'p' in noaug_target:
                        tmp_loss = (lambda_noaug * noaug_samplew * noaug_criterion(aug_weight,noaug_w_target)).mean()
                        noaug_loss += tmp_loss
                        print('prob loss: ',tmp_loss.detach().item())
                    if torch.is_tensor(noaug_loss):
                        noaug_reg_sum += noaug_loss.detach().mean().item()
                    #print('noaug regular sum: ',noaug_reg_sum)
                else:
                    noaug_loss = 0
                #tea
                if teacher_model==None:
                    sim_model = gf_model
                else:
                    sim_model = teacher_model.module
                #sim mix if need, calculate loss
                augsear_loss, logits_search, each_aug_loss = mix_func(gf_model,mixed_features,aug_weights,sim_criterion,target_search,multilabel)
                origin_embed = sim_model.extract_features(input_search, seq_len, pool=True)
                origin_logits = sim_model.classify(origin_embed)
                ori_loss = cuc_loss(origin_logits,target_search,sim_criterion,multilabel)
                #ori_loss = ori_loss.mean() #!to assert loss mean reduce
                #output pred
                if not multilabel:
                    if mix_type=='loss': #tmp for visualize
                        if len(logits_search.shape)==4:
                            logits_search_avg = logits_search.mean(dim=(1,2))
                        else:
                            logits_search_avg = logits_search.mean(dim=1)
                        soft_out = softmax_m(logits_search_avg).detach().cpu() #(bs,n_class) for embed or output mix
                        soft_all = softmax_m(logits_search.reshape(-1,n_class)).detach().cpu().reshape(*logits_search.shape) #(bs,keep_len,n_ops)
                        origin_embed_out = origin_embed.detach().cpu()
                        each_aug_loss = each_aug_loss.detach().cpu()
                        #print('soft all ',soft_all.shape)
                        #print('each_aug_loss ',each_aug_loss.shape)
                        for i,t in enumerate(target_search.data.view(-1)):
                            sea_output_matrix[t.long()] += soft_out[i]
                            sea_embed_matrix[t.long(),:] += origin_embed_out[i] #!!! 11/14 add, use origin
                            sea_embed_count[t.long(),0] += 1
                            for aug_idx in range(n_ops):
                                aug_output_matrix[t.long(),aug_idx] += soft_all[i,aug_idx]
                                aug_loss_mat[aug_idx] += each_aug_loss[i,aug_idx]
                                aug_class_loss[t.long(),aug_idx] += each_aug_loss[i,aug_idx]
                    else:
                            soft_out = softmax_m(logits_search).detach().cpu() #(bs,n_class) for embed or output mix
                            origin_embed_out = origin_embed.detach().cpu()
                            for i,t in enumerate(target_search.data.view(-1)):
                                sea_output_matrix[t.long()] += soft_out[i]
                                sea_embed_matrix[t.long(),:] += origin_embed_out[i] #!!! 11/14 add, use origin
                                sea_embed_count[t.long(),0] += 1

                #similar reweight?
                aug_search_loss += augsear_loss.detach().mean().item()
                ori_search_loss += ori_loss.detach().mean().item()
                loss = sim_loss_func(ori_loss,augsear_loss,**add_kwargs)
                #print(loss.shape,loss) #!tmp
                if torch.is_tensor(class_weight): #class_weight: (n_class), w_sample: (bs)
                    class_weight = class_weight.cuda()
                    if not multilabel:
                        w_sample = class_weight[target_search].detach()
                    else: #class_weight:(1,n_class), target_search:(bs,n_class)
                        w_sample = (class_weight.view(1,n_class) * target_search).sum(1)
                    print('w_sample: ',w_sample.shape)
                    tmp = w_sample * loss
                    print('sim loss: ',tmp)
                    loss = tmp.mean() 
                elif sim_reweight: #reweight part, a,b = ?
                    p_orig = origin_logits.softmax(dim=1)[torch.arange(search_bs), target_search].detach()
                    p_aug = logits_search.softmax(dim=1)[torch.arange(search_bs), target_search].clone().detach()
                    w_aug = torch.sqrt((1.0 - p_orig)) #a=0.5,b=0.5
                    if w_aug.sum() > 0:
                        w_aug /= (w_aug.mean().detach() + 1e-6)
                    else:
                        w_aug = 1
                    loss = (w_aug * loss).mean()
                    #print('Similar loss:', loss.detach().item())
                else:
                    loss = loss.mean()
                    #print('Similar loss:', loss.detach().item())
                loss = loss * lambda_sim + noaug_loss
                #extra losses
                for e_criterion in extra_criterions:
                    if e_criterion.loss_target=='output': #for class distance loss target
                        e_loss = e_criterion(logits_search,target_search,sim_targets=origin_logits)
                    elif e_criterion.loss_target=='embed': 
                        e_loss = e_criterion(mixed_features,target_search,sim_targets=origin_embed) #using mix_fearures as loss
                    elif e_criterion.loss_target=='policy': 
                        e_loss = e_criterion(aug_policy,target_search,sim_targets=origin_embed) #using mix_fearures as loss
                    else:
                        print('Unknown loss target: ',e_criterion.loss_target)
                        raise
                    if torch.is_tensor(e_loss):
                        loss += e_loss.mean()
                        ex_losses[str(e_criterion.__class__.__name__)] += e_loss.detach().item()
                        print('Extra class distance loss:', e_loss.detach().item())
                #!!!10/13 bug fix, tmp *4 for plr!!!
                loss = loss * 4 / search_round
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)): #!tmp
                    print('Loss error: ',loss.detach().item())
                    print('ori_loss',ori_loss)
                    print('augsear_loss',augsear_loss)
                    print('target_search',target_search)
                    print('origin_logits',origin_logits)
                    print('logits_search',logits_search)
                    print('noaug_loss',noaug_loss)

                loss.backward()
                adaptive_loss += loss.detach().item()
                search_total += 1
                input_search_list.append(input_search.detach())
                seq_len_list.append(seq_len.detach())
                target_search_list.append(target_search.detach())
                #torch.cuda.empty_cache()'''
            #accumulation update
            nn.utils.clip_grad_norm_(adaaug.h_model.parameters(), grad_clip)
            h_optimizer.step()
            #  log policy
            gf_optimizer.zero_grad()
            h_optimizer.zero_grad()
            input_search_list = torch.cat(input_search_list,dim=0)
            seq_len_list = torch.cat(seq_len_list,dim=0)
            target_search_list = torch.cat(target_search_list,dim=0)
            if class_adaptive:
                policy_y_list = torch.cat(policy_y_list,dim=0)
            else:
                policy_y_list = None
            policy = adaaug.add_history(input_search_list, seq_len_list, target_search_list,y=policy_y_list)
            #back to policy
            #policy_out = torch.cat(policy,dim=1).detach().cpu() #[mag, weight] is same with h_model output
            policy_out = select_npolicy(policy,policy_dist=policy_dist).detach().cpu()
            sea_policy_matrix += policy_out #sum of policy for each class

        exploration_time = time.time() - timer
        torch.cuda.empty_cache()
        #log
        global_step = epoch * len(train_queue) + step
        if global_step % args.report_freq == 0:
            logging.info('  |train %03d %e %f %f | %.3f + %.3f s', global_step,
                objs.avg, top1.avg, top5.avg, exploitation_time, exploration_time)
    #visualize
    if visualize:
        input_search, seq_len, target_search = next(iter(search_queue))
        input_search = input_search.float().cuda()
        target_search = target_search.cuda()
        policy_y = None
        if class_adaptive: #target to onehot
            if not multilabel:
                policy_y = nn.functional.one_hot(target_search, num_classes=n_class).cuda().float()
            else:
                policy_y = target_search.cuda().float()
        adaaug.visualize_result(input_search, seq_len,policy_y,target_search)
        exit()
    #class-wise & Total
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/torch.clamp(confusion_matrix.sum(1),min=1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / torch.clamp(confusion_matrix.sum(),min=1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
        ptype = 'acc'
        logging.info(f'Epoch train: loss={objs.avg} top1acc={top1.avg} top5acc={top5.avg}')
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom_cw2 = utils.mAP_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom_cw2.mean()
        logging.info('Epoch train: loss=%e macroAUROC=%f', objs.avg, perfrom)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Epoch train: loss=%e macroAP=%f', objs.avg, perfrom2)
        logging.info('class-wise AP: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw2])+']')
        ptype = 'auroc'
    #table dic
    table_dic = {}
    table_dic['search_output'] = (sea_output_matrix / torch.clamp(sea_output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic['train_output'] = (tr_output_matrix / torch.clamp(tr_output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic['train_embed'] = tr_embed_matrix / torch.clamp(tr_embed_count.float(),min=1e-9)
    table_dic['train_policy'] = tr_policy_matrix / tr_embed_count.sum(0)
    table_dic['search_embed'] = sea_embed_matrix / torch.clamp(sea_embed_count.float(),min=1e-9)
    table_dic['search_policy'] = sea_policy_matrix / torch.clamp(sea_embed_count.float(),min=1e-9)
    #sample
    print('sample policy class 0: ',table_dic['search_policy'][0])
    #average policy for all train+search data
    table_dic['train_confusion'] = confusion_matrix
    #tmp for aug loss visual
    if mix_type=='loss':
        table_dic['aug_output'] = aug_output_matrix / torch.clamp(aug_output_matrix.sum(dim=-1,keepdim=True),min=1e-9)
        table_dic['aug_loss'] = aug_loss_mat / torch.sum(sea_embed_count.float())
        table_dic['class_aug_loss'] = aug_class_loss / torch.clamp(sea_embed_count.float(),min=1e-9)
        print('aug_output: ',table_dic['aug_output'].shape) #tmp
        print('aug_loss: ',table_dic['aug_loss'].shape) #tmp
        print('class_aug_loss: ',table_dic['class_aug_loss'].shape) #tmp
        print('aug_output: ',table_dic['aug_output']) #tmp
        print('aug_loss: ',table_dic['aug_loss']) #tmp
        print('class_aug_loss: ',table_dic['class_aug_loss']) #tmp
    #wandb dic
    out_dic = {}
    out_dic['train_loss'] = objs.avg
    for k in sorted(ex_losses):
        out_dic[k] = ex_losses[k] / search_total
    if epoch>= warmup_epoch:
        out_dic['adaptive_loss'] = adaptive_loss * search_round / (search_total*4)
        out_dic['aug_sear_loss'] = aug_search_loss / search_total
        out_dic['ori_sear_loss'] = ori_search_loss / search_total
        if difficult_aug:
            out_dic['difficult_loss'] = difficult_loss * search_round / (search_total*4)
            out_dic['reweight_sum'] = re_weights_sum / search_total
            out_dic['aug_diff_loss'] = aug_diff_loss / search_total
            out_dic['ori_diff_loss'] = ori_diff_loss / search_total
            out_dic['search_loss'] = out_dic['adaptive_loss']+out_dic['difficult_loss']
        else:
            out_dic['search_loss'] = out_dic['adaptive_loss']
        if use_noaug_reg:
            out_dic['noaug_reg'] = noaug_reg_sum / search_total
        
    out_dic[f'train_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'train_{ptype}_c{i}'] = e_c
    if multilabel: #addition
        add_type = 'ap'
        out_dic[f'train_{add_type}_avg'] = perfrom2
        for i,e_c in enumerate(perfrom_cw2):
            out_dic[f'train_{add_type}_c{i}'] = e_c
        if map_select:
            return perfrom2, objs.avg, out_dic, table_dic

    return perfrom, objs.avg, out_dic, table_dic