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

def train(args, train_queue, model, criterion, optimizer,scheduler, epoch, grad_clip, adaaug, multilabel=False,n_class=10,
        difficult_aug=False,reweight=True,lambda_aug = 1.0,class_adaptive=False,map_select=False,visualize=False,training=True,
        teach_rew=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    confusion_matrix = torch.zeros(n_class,n_class)
    tr_output_matrix = torch.zeros(n_class,n_class).float()
    softmax_m = nn.Softmax(dim=1)
    preds = []
    targets = []
    total = 0
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
        #teach reweight
        if teach_rew!=None:
            teach_ori_logits = model(input, seq_len)
            p_aug = logits.softmax(dim=1)[torch.arange(batch_size), target].clone().detach()
            t_aug = teach_ori_logits.softmax(dim=1)[torch.arange(batch_size), target].clone().detach()
            w_aug = torch.sqrt(t_aug * torch.clamp(t_aug - p_aug, min=0)) + 1 #a=0.5,b=0.5 ???
            w_aug /= (w_aug.mean().detach() + 1e-9)
            aug_loss = (w_aug * aug_loss).mean()

        loss = ori_loss + aug_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if training:
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
            soft_out = softmax_m(logits).detach().cpu()
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            for i,t in enumerate(target.data.view(-1)):
                tr_output_matrix[t.long(),:] += soft_out[i]
        else:
            predicted = torch.sigmoid(logits.data)
        preds.append(predicted.cpu().detach())
        targets.append(target.cpu().detach())
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
    #class-wise & Total
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / (confusion_matrix.sum()+1e-9)
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
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw2])+']')
        ptype = 'auroc'
    #table dic
    table_dic = {}
    table_dic['train_output'] = (tr_output_matrix / torch.clamp(tr_output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic['train_confusion'] = confusion_matrix
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
                soft_out = softmax_m(logits).detach().cpu()
                for t, p in zip(target.data.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                for i,t in enumerate(target.data.view(-1)):
                    output_matrix[t.long(),:] += soft_out[i]
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
            else:
                predicted = torch.sigmoid(logits.data)
            preds.append(predicted.cpu().detach())
            targets.append(target.cpu().detach())
    #table dic
    table_dic = {}
    table_dic[f'{mode}_output'] = (output_matrix / torch.clamp(output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic[f'{mode}_confusion'] = confusion_matrix
    #class-wise
    if not multilabel:
        cw_acc = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / (confusion_matrix.sum()+1e-9)
        logging.info('Overall Acc: %f',nol_acc)
        perfrom = top1.avg
        perfrom_cw = cw_acc
        perfrom2 = top5.avg
        ptype = 'acc'
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
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

def cuc_loss(logits,target,criterion,multilabel,**kwargs):
    if multilabel:
        loss = criterion(logits, target.float(),**kwargs)
    else:
        loss = criterion(logits, target.long(),**kwargs)
    return loss

def embed_mix(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel):
    aug_logits = gf_model.classify(mixed_features) 
    aug_loss = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
    return aug_loss, aug_logits

def loss_mix(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel):
    # mixed_features=(batch, n_ops,keep_lens, n_hidden) or (batch, keep_lens, n_hidden) or (batch, n_ops, n_hidden)
    target_trsearch = torch.unsqueeze(target_trsearch,dim=1) #(bs,1) !!!tmp not for multilabel
    if len(aug_weights)==2: #(batch, n_ops,keep_lens, n_hidden)
        batch, n_ops,keep_lens, n_hidden = mixed_features.shape
        weights, keeplen_ws = aug_weights[0], aug_weights[1]
        target_trsearch = target_trsearch.expand(-1,n_ops*keep_lens).reshape(batch*n_ops*keep_lens)
        mixed_features = mixed_features.reshape(batch*n_ops*keep_lens,n_hidden)
        aug_logits = gf_model.classify(mixed_features)
        aug_loss = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
        aug_loss = aug_loss.reshape(batch, n_ops,keep_lens).permute(0,2,1)
        # weights = (bs,n_ops), aug_loss = (batch,keep_lens,n_ops)
        aug_loss = [w.matmul(feat) for w, feat in zip(weights, aug_loss)] #[(keep_lens)]
        aug_loss = torch.stack([len_w.matmul(feat) for len_w,feat in zip(keeplen_ws,aug_loss)], dim=0) #[(1)]
        #print('Loss mix aug_loss: ',aug_loss.shape,aug_loss)
        aug_loss = aug_loss.mean()
        aug_logits = aug_logits.reshape(batch, n_ops,keep_lens,-1).mean(dim=(1,2))
    else:
        batch, n_param, n_hidden = mixed_features.shape
        weights = aug_weights[0]
        target_trsearch = target_trsearch.expand(-1,n_param).reshape(batch*n_param)
        mixed_features = mixed_features.reshape(batch*n_param,n_hidden)
        aug_logits = gf_model.classify(mixed_features)
        aug_loss = cuc_loss(aug_logits,target_trsearch,adv_criterion,multilabel)
        aug_loss = aug_loss.reshape(batch, n_param)
        aug_loss = torch.stack([w.matmul(feat) for w, feat in zip(weights, aug_loss)], dim=0) #[(1)]
        #print('Loss mix aug_loss: ',aug_loss.shape,aug_loss)
        aug_loss = aug_loss.mean()
        aug_logits = aug_logits.reshape(batch, n_param,-1).mean(dim=1)
    #print('Loss mix aug_logits: ',aug_logits.shape,aug_logits)
    return aug_loss, aug_logits

def search_train(args, train_queue, search_queue, tr_search_queue, gf_model, adaaug, criterion, gf_optimizer,scheduler,
            grad_clip, h_optimizer, epoch, search_freq,search_round=1,search_repeat=1, multilabel=False,n_class=10,
            difficult_aug=False,same_train=False,reweight=True,sim_reweight=False,mix_type='embed', warmup_epoch = 0
            ,lambda_sim = 1.0,lambda_aug = 1.0,loss_type='minus',lambda_noaug = 0,train_perfrom = 0.0,
            class_adaptive=False,adv_criterion=None,sim_criterion=None,extra_criterions=[],teacher_model=None,map_select=False,
            visualize=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    confusion_matrix = torch.zeros(n_class,n_class)
    tr_output_matrix = torch.zeros(n_class,n_class).float()
    sea_output_matrix = torch.zeros(n_class,n_class).float()
    softmax_m = nn.Softmax(dim=1)
    
    print(loss_type)
    print(adv_criterion)
    if adv_criterion==None:
        adv_criterion = criterion
    if sim_criterion==None:
        sim_criterion = criterion
    noaug_criterion = nn.CrossEntropyLoss().cuda()
    diff_update_w = True
    sim_loss_func = ab_loss
    if loss_type=='minus':
        diff_loss_func = minus_loss
    elif loss_type=='relative':
        diff_loss_func = relative_loss
        sim_loss_func = rel_loss
    elif loss_type=='relativediff':
        diff_loss_func = relative_loss
        diff_update_w = False
    elif loss_type=='adv':
        diff_loss_func = adv_loss
    else:
        print('Unknown loss type for policy training')
        print(loss_type)
        print(adv_criterion)
        raise
    mix_feature = False
    if mix_type=='embed':
        mix_feature = True
        mix_func = embed_mix
        print('Embed mix type')
    elif mix_type=='loss':
        mix_func = loss_mix
        print('Loss mix type')
    else:
        print('Unknown mix type')
        raise
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
            aug_images = adaaug(input, seq_len, mode='exploit', y=policy_y, policy_apply=policy_apply)
        else:
            aug_images = input
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
            soft_out = softmax_m(logits).detach().cpu()
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            for i,t in enumerate(target.data.view(-1)):
                tr_output_matrix[t.long(),:] += soft_out[i]
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
                    origin_logits = gf_model(input_trsearch, seq_len)
                    policy_y = None
                    if class_adaptive: #target to onehot
                        if not multilabel:
                            policy_y = nn.functional.one_hot(target_trsearch, num_classes=n_class).cuda().float()
                        else:
                            policy_y = target_trsearch.cuda().float()
                    mixed_features, aug_weights = adaaug(input_trsearch, seq_len, mode='explore',mix_feature=mix_feature,y=policy_y,update_w=diff_update_w)
                    aug_loss, aug_logits = mix_func(gf_model,mixed_features,aug_weights,adv_criterion,target_trsearch,multilabel)
                    ori_loss = cuc_loss(origin_logits,target_trsearch,adv_criterion,multilabel).mean() #!to assert loss mean reduce
                    aug_diff_loss += aug_loss.detach().item()
                    ori_diff_loss += ori_loss.detach().item()
                    loss_prepolicy = diff_loss_func(ori_loss=ori_loss,aug_loss=aug_loss,lambda_aug=lambda_aug)
                    if reweight: #reweight part, a,b = ?
                        p_orig = origin_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].detach()
                        p_aug = aug_logits.softmax(dim=1)[torch.arange(batch_size), target_trsearch].clone().detach()
                        w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0)) #a=0.5,b=0.5
                        re_weights_sum += w_aug.sum().detach().item()
                        if w_aug.sum() > 0:
                            w_aug /= (w_aug.mean().detach() + 1e-6)
                        else:
                            w_aug = 1
                        loss_policy = (w_aug * loss_prepolicy).mean()
                    else:
                        loss_policy = loss_prepolicy.mean()
                    #!!!10/13 bug fix!!! ,tmp*4 for same plr
                    loss_policy = loss_policy * 4 / search_round
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
                #weight regular to NOAUG
                aug_weight = aug_weights[0] # (bs, n_ops), 0 is NOAUG
                noaug_target = torch.zeros(target_search.shape).cuda().long()
                if lambda_noaug>0: #need test
                    noaug_loss = lambda_noaug * (1.0 - train_perfrom) * noaug_criterion(aug_weight,noaug_target)
                    noaug_reg_sum += noaug_loss.detach().item()
                else:
                    noaug_loss = 0
                #tea
                if teacher_model==None:
                    sim_model = gf_model
                else:
                    sim_model = teacher_model.module
                #sim mix if need, calculate loss
                loss, logits_search = mix_func(gf_model,mixed_features,aug_weights,sim_criterion,target_search,multilabel)
                origin_logits = sim_model(input_search, seq_len)
                ori_loss = cuc_loss(origin_logits,target_search,sim_criterion,multilabel)
                ori_loss = ori_loss.mean() #!to assert loss mean reduce
                #output pred
                soft_out = softmax_m(logits_search).detach().cpu() #(bs,n_class)
                for i,t in enumerate(target_search.data.view(-1)):
                    sea_output_matrix[t.long()] += soft_out[i]

                #similar reweight?
                aug_search_loss += loss.detach().item()
                ori_search_loss += ori_loss.detach().item()
                loss = sim_loss_func(ori_loss,loss)
                if sim_reweight: #reweight part, a,b = ?
                    p_orig = origin_logits.softmax(dim=1)[torch.arange(search_bs), target_search].detach()
                    p_aug = logits_search.softmax(dim=1)[torch.arange(search_bs), target_search].clone().detach()
                    w_aug = torch.sqrt((1.0 - p_orig)) #a=0.5,b=0.5
                    if w_aug.sum() > 0:
                        w_aug /= (w_aug.mean().detach() + 1e-6)
                    else:
                        w_aug = 1
                    loss = (w_aug * loss).mean()
                else:
                    loss = loss.mean()
                loss = loss * lambda_sim + noaug_loss
                #extra losses
                for e_criterion in extra_criterions:
                    e_loss = e_criterion(logits_search,target_search)
                    if torch.is_tensor(e_loss):
                        loss += e_loss
                        ex_losses[str(e_criterion.__class__.__name__)] += e_loss.detach().item()
                #!!!10/13 bug fix, tmp *4 for plr!!!
                loss = loss * 4 / search_round
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
            else:
                policy_y_list = None
            adaaug.add_history(input_search_list, seq_len_list, target_search_list,y=policy_y_list)
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
        cw_acc = 100 * confusion_matrix.diag()/(confusion_matrix.sum(1)+1e-9)
        logging.info('class-wise Acc: ' + str(cw_acc))
        nol_acc = 100 * confusion_matrix.diag().sum() / (confusion_matrix.sum()+1e-9)
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
    #print(sea_output_matrix.sum()) #! tmp
    #print(sea_output_matrix.sum(dim=1))
    #print(sea_output_matrix) #! tmp
    table_dic['search_output'] = (sea_output_matrix / torch.clamp(sea_output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic['train_output'] = (tr_output_matrix / torch.clamp(tr_output_matrix.sum(dim=1,keepdim=True),min=1e-9))
    table_dic['train_confusion'] = confusion_matrix
    #print(table_dic['search_output'].sum()) #! tmp
    #print(table_dic['search_output']) #! tmp
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
        if lambda_noaug>0:
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
    softmax_m = nn.Softmax(dim=1)
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
                soft_out = softmax_m(logits).detach().cpu()
                for t, p in zip(target.data.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                for i,t in enumerate(target.data.view(-1)):
                    output_matrix[t.long(),:] += soft_out[i]
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
        ptype = 'acc'
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom_cw2 = utils.mAP_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom_cw2.mean()
        logging.info('Epoch %s: loss=%e macroAUROC=%f', mode, objs.avg, perfrom)
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Epoch %s: loss=%e macroAP=%f',mode, objs.avg, perfrom2)
        logging.info('class-wise AP: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw2])+']')
        ptype = 'auroc'
    #table dic
    table_dic = {}
    table_dic[f'{mode}_output'] = (output_matrix / torch.clamp(output_matrix.sum(dim=1,keepdim=True),1e-9))
    table_dic[f'{mode}_confusion'] = confusion_matrix
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