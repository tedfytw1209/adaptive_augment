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


def train(args, train_queue, model, criterion, optimizer,scheduler, epoch, grad_clip, adaaug, multilabel=False,n_class=10,
        difficult_aug=False,reweight=True,lambda_aug = 1.0,class_adaptive=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    confusion_matrix = torch.zeros(n_class,n_class)
    preds = []
    targets = []
    total = 0
    for step, (input, seq_len, target) in enumerate(train_queue):
        input = input.float().cuda()
        target = target.cuda(non_blocking=True)
        #  get augmented training data from adaaug
        if class_adaptive: #target to onehot
            policy_y = nn.functional.one_hot(target, num_classes=n_class).cuda().float()
        else:
            policy_y = None
        aug_images = adaaug(input, seq_len, mode='exploit',y=policy_y)
        model.train()
        optimizer.zero_grad()
        logits = model(aug_images, seq_len)
        if multilabel:
            aug_loss = criterion(logits, target.float())
        else:
            aug_loss = criterion(logits, target.long())
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
        loss = ori_loss + aug_loss
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
        if step == 0:
            adaaug.add_history(input, seq_len, target,y=policy_y)
        
        # Accuracy / AUROC
        if not multilabel:
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        else:
            predicted = torch.sigmoid(logits.data)
        preds.append(predicted.cpu().detach())
        targets.append(target.cpu().detach())
    #Total
    if not multilabel:
        perfrom = 100 * top1.avg
        logging.info('Epoch train: loss=%e top1acc=%f top5acc=%f', objs.avg, top1.avg, top5.avg)
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        try:
            perfrom = 100 * roc_auc_score(targets_np, preds_np,average='macro')
        except Exception as e:
            nan_count = np.sum(np.isnan(preds_np))
            inf_count = np.sum(np.isinf(preds_np))
            print('predict nan, inf count: ',nan_count,inf_count)
            raise e
        logging.info('Epoch train: loss=%e macroAUROC=%f', objs.avg, perfrom)
    #class-wise
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / (confusion_matrix.sum()+1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Overall AUROC: %f',perfrom)
    #wandb dic
    out_dic = {}
    out_dic[f'train_loss'] = objs.avg
    if multilabel:
        ptype = 'auroc'
    else:
        ptype = 'acc'
    out_dic[f'train_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'train_{ptype}_c{i}'] = e_c
    
    return perfrom, objs.avg, out_dic

def infer(valid_queue, model, criterion, multilabel=False, n_class=10,mode='test'):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    confusion_matrix = torch.zeros(n_class,n_class)
    preds = []
    targets = []
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
                for t, p in zip(target.data.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
            else:
                predicted = torch.sigmoid(logits.data)
            preds.append(predicted.cpu().detach())
            targets.append(target.cpu().detach())

    #class-wise
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / (confusion_matrix.sum()+1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
        perfrom2 = top5.avg
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Overall AUROC: %f',perfrom)
    #wandb dic
    out_dic = {}
    out_dic[f'{mode}_loss'] = objs.avg
    if multilabel:
        ptype = 'auroc'
    else:
        ptype = 'acc'
    out_dic[f'{mode}_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'{mode}_{ptype}_c{i}'] = e_c
    
    return perfrom, objs.avg, perfrom2, objs.avg, out_dic

def relative_loss(ori_loss, aug_loss, lambda_aug):
    return lambda_aug * (ori_loss.detach() / aug_loss)
def minus_loss(ori_loss, aug_loss, lambda_aug):
    return -1 * lambda_aug * aug_loss
def adv_loss(ori_loss, aug_loss, lambda_aug):
    return lambda_aug * aug_loss

def ab_loss(ori_loss, aug_loss):
    return aug_loss
def rel_loss(ori_loss, aug_loss):
    return aug_loss / ori_loss.detach()

def search_train(args, train_queue, search_queue, tr_search_queue, gf_model, adaaug, criterion, gf_optimizer,scheduler,
            grad_clip, h_optimizer, epoch, search_freq,search_round=1, multilabel=False,n_class=10,
            difficult_aug=False,same_train=False,reweight=True,mix_feature=True,lambda_aug = 1.0,loss_type='minus',
            class_adaptive=False,adv_criterion=None,sim_criterion=None,teacher_model=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    confusion_matrix = torch.zeros(n_class,n_class)
    
    print(loss_type)
    print(adv_criterion)
    if adv_criterion==None:
        adv_criterion = criterion
    if sim_criterion==None:
        sim_criterion = criterion
    
    sim_loss_func = ab_loss
    if loss_type=='minus':
        diff_loss_func = minus_loss
    elif loss_type=='relative':
        diff_loss_func = relative_loss
        sim_loss_func = rel_loss
    elif loss_type=='adv':
        diff_loss_func = adv_loss
    else:
        print('Unknown loss type for policy training')
        print(loss_type)
        print(adv_criterion)
        raise
    preds = []
    targets = []
    total = 0
    difficult_loss, adaptive_loss, search_total = 0, 0, 0
    for step, (input, seq_len, target) in enumerate(train_queue):
        input = input.float().cuda() #(batch,sed_len,channel)
        target = target.cuda()
        if class_adaptive: #target to onehot
            policy_y = nn.functional.one_hot(target, num_classes=n_class).cuda().float()
        else:
            policy_y = None
        # exploitation
        timer = time.time()
        aug_images = adaaug(input, seq_len, mode='exploit',y=policy_y)
        aug_images = aug_images.cuda()
        gf_model.train()
        gf_optimizer.zero_grad()
        logits = gf_model(aug_images, seq_len)
        if multilabel:
            aug_loss = criterion(logits, target.float())
        else:
            aug_loss = criterion(logits, target.long())
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
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        else:
            predicted = torch.sigmoid(logits)
        
        #ACC-AUROC
        preds.append(predicted.cpu().detach())
        targets.append(target.cpu().detach())
        exploitation_time = time.time() - timer

        # exploration
        timer = time.time()
        if step % search_freq == 0:
            #difficult, train input, target
            gf_optimizer.zero_grad()
            h_optimizer.zero_grad()
            input_search_list,seq_len_list,target_search_list,policy_y_list = [],[],[],[]
            for r in range(search_round): #grad accumulation
                if difficult_aug:
                    input_trsearch, seq_len, target_trsearch = next(iter(tr_search_queue))
                    input_trsearch = input_trsearch.float().cuda()
                    target_trsearch = target_trsearch.cuda()
                    batch_size = target_trsearch.shape[0]
                    origin_logits = gf_model(input_trsearch, seq_len)
                    if class_adaptive: #target to onehot
                        policy_y = nn.functional.one_hot(target_trsearch, num_classes=n_class).cuda().float()
                    else:
                        policy_y = None
                    mixed_features = adaaug(input_trsearch, seq_len, mode='explore',mix_feature=mix_feature,y=policy_y)
                    aug_logits = gf_model.classify(mixed_features) 
                    if multilabel:
                        aug_loss = adv_criterion(aug_logits, target_trsearch.float())
                        ori_loss = adv_criterion(origin_logits, target_trsearch.float())
                    else:
                        aug_loss = adv_criterion(aug_logits, target_trsearch.long())
                        ori_loss = adv_criterion(origin_logits, target_trsearch.long())
                    loss_prepolicy = diff_loss_func(ori_loss=ori_loss,aug_loss=aug_loss,lambda_aug=lambda_aug)
                    if reweight: #reweight part, a,b = ?
                        p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].detach()
                        p_aug = aug_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].clone().detach()
                        w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
                        if w_aug.sum() > 0:
                            w_aug /= (w_aug.mean().detach() + 1e-6)
                        else:
                            w_aug = 1
                        loss_policy = (w_aug * loss_prepolicy).mean()
                    else:
                        loss_policy = loss_prepolicy.mean()
                    
                    loss_policy.backward()
                    #h_optimizer.step() wait till validation set
                    difficult_loss += loss_policy.detach().item()
                    torch.cuda.empty_cache()
                #similar
                input_search, seq_len, target_search = next(iter(search_queue))
                input_search = input_search.float().cuda()
                target_search = target_search.cuda()
                if class_adaptive: #target to onehot
                    policy_y = nn.functional.one_hot(target_search, num_classes=n_class).cuda().float()
                    policy_y_list.append(policy_y)
                else:
                    policy_y = None
                    policy_y_list = None
                mixed_features = adaaug(input_search, seq_len, mode='explore',y=policy_y)
                #tea
                if teacher_model==None:
                    sim_model = gf_model
                else:
                    sim_model = teacher_model.module
                logits_search = sim_model.classify(mixed_features)
                origin_logits = sim_model(input_search, seq_len)
                #calculate loss
                if multilabel:
                    loss = sim_criterion(logits_search, target_search.float())
                    ori_loss = sim_criterion(origin_logits, target_search.float())
                else:
                    loss = sim_criterion(logits_search, target_search.long())
                    ori_loss = sim_criterion(origin_logits, target_search.long())
                loss = sim_loss_func(ori_loss,loss)
                loss.backward()

                adaptive_loss += loss.detach().item()
                search_total += 1
                input_search_list.append(input_search.detach())
                seq_len_list.append(seq_len.detach())
                target_search_list.append(target_search.detach())
                torch.cuda.empty_cache()
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
            adaaug.add_history(input_search_list, seq_len_list, target_search_list,y=policy_y_list)
        exploration_time = time.time() - timer
        torch.cuda.empty_cache()
        #log
        global_step = epoch * len(train_queue) + step
        if global_step % args.report_freq == 0:
            logging.info('  |train %03d %e %f %f | %.3f + %.3f s', global_step,
                objs.avg, top1.avg, top5.avg, exploitation_time, exploration_time)

    #Total
    if not multilabel:
        perfrom = top1.avg
        logging.info('Epoch train: loss=%e top1acc=%f top5acc=%f', objs.avg, top1.avg, top5.avg)
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        try:
            perfrom = 100 * roc_auc_score(targets_np, preds_np,average='macro')
        except Exception as e:
            nan_count = np.sum(np.isnan(preds_np))
            inf_count = np.sum(np.isinf(preds_np))
            print('predict nan, inf count: ',nan_count,inf_count)
            raise e
        logging.info('Epoch train: loss=%e macroAUROC=%f', objs.avg, perfrom)
    #class-wise
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / (confusion_matrix.sum()+1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Overall AUROC: %f',perfrom)
    
    #wandb dic
    out_dic = {}
    out_dic['train_loss'] = objs.avg
    out_dic['adaptive_loss'] = adaptive_loss / search_total
    out_dic['difficult_loss'] = difficult_loss / search_total
    out_dic['search_loss'] = out_dic['adaptive_loss']+out_dic['difficult_loss']
    if multilabel:
        ptype = 'auroc'
    else:
        ptype = 'acc'
    out_dic[f'train_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'train_{ptype}_c{i}'] = e_c

    return perfrom, objs.avg, out_dic


def search_infer(valid_queue, gf_model, criterion, multilabel=False, n_class=10,mode='test'):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    gf_model.eval()
    confusion_matrix = torch.zeros(n_class,n_class)
    preds = []
    targets = []
    total = 0
    with torch.no_grad():
        for input, seq_len, target in valid_queue:
            input = input.float().cuda()
            target = target.cuda()

            logits = gf_model(input,seq_len)
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
                for t, p in zip(target.data.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
            else:
                predicted = torch.sigmoid(logits.data)
            preds.append(predicted.cpu().detach())
            targets.append(target.cpu().detach())
    #class-wise
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        logging.info(f'{mode} class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / (confusion_matrix.sum()+1e-9)
        logging.info('%s Overall Acc: %f',mode,nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        logging.info(f'{mode} class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('%s Overall AUROC: %f',mode,perfrom)
    #wandb dic
    out_dic = {}
    out_dic[f'{mode}_loss'] = objs.avg
    if multilabel:
        ptype = 'auroc'
    else:
        ptype = 'acc'
    out_dic[f'{mode}_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'{mode}_{ptype}_c{i}'] = e_c
    
    return perfrom, objs.avg, out_dic
    #return top1.avg, objs.avg