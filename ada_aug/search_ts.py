from __future__ import print_function
from __future__ import absolute_import
from multiprocessing import reduction

import os
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

from adaptive_augmentor import AdaAug_TS
from networks import get_model_tseries
from networks.projection import Projection_TSeries
from config import get_search_divider
from dataset import get_ts_dataloaders, get_num_class, get_label_name, get_dataset_dimension,get_num_channel
from operation_tseries import TS_OPS_NAMES,ECG_OPS_NAMES,TS_ADD_NAMES,MAG_TEST_NAMES,NOMAG_TEST_NAMES
import wandb

parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--dataroot', type=str, default='./', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
parser.add_argument('--labelgroup', default='')
parser.add_argument('--test_size', type=float, default=0.2, help='test size')
parser.add_argument('--search_size', type=float, default=0.5, help='test size')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.400, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--multilabel', action='store_true', default=False, help='using multilabel default False')
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
parser.add_argument('--default_split', action='store_true', help='use dataset deault split')
parser.add_argument('--proj_learning_rate', type=float, default=1e-2, help='learning rate for h')
parser.add_argument('--proj_weight_decay', type=float, default=1e-3, help='weight decay for h]')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--use_parallel', type=bool, default=False, help="use data parallel default False")
parser.add_argument('--model_name', type=str, default='wresnet40_2', help="mode _name")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--k_ops', type=int, default=1, help="number of augmentation applied during training")
parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
parser.add_argument('--search_freq', type=float, default=1, help='exploration frequency')
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of hidden layer in augmentation policy projection")
parser.add_argument('--valselect', action='store_true', default=False, help='use valid select')

args = parser.parse_args()
debug = True if args.save == "debug" else False
args.save = '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save)
if debug:
    args.save = os.path.join('debug', args.save)
else:
    args.save = os.path.join('search', args.dataset, args.save)
utils.create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    utils.reproducibility(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    #wandb
    if args.k_ops>0:
        Aug_type = 'AdaAug'
    else:
        Aug_type = 'NOAUG'
    experiment_name = f'{Aug_type}_search_vselect_{args.dataset}{args.labelgroup}_{args.model_name}_e{args.epochs}_lr{args.learning_rate}'
    run_log = wandb.init(config=args, 
                  project='AdaAug',
                  group=experiment_name,
                  name=experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)

    #  dataset settings
    n_channel = get_num_channel(args.dataset)
    n_class = get_num_class(args.dataset,args.labelgroup)
    sdiv = get_search_divider(args.model_name)
    class2label = get_label_name(args.dataset, args.dataroot)
    multilabel = args.multilabel
    train_queue, valid_queue, search_queue, test_queue = get_ts_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        args.dataroot, args.cutout, args.cutout_length,
        split=args.train_portion, split_idx=0, target_lb=-1,
        search=True, search_divider=sdiv,search_size=args.search_size,
        test_size=args.test_size,multilabel=args.multilabel,default_split=args.default_split,
        labelgroup=args.labelgroup)
    
    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'  |total: {len(train_queue.dataset)}')
    logging.info(f'  |train: {len(train_queue)*args.batch_size}')
    logging.info(f'  |valid: {len(valid_queue)*args.batch_size}')
    logging.info(f'  |search: {len(search_queue)*sdiv}')

    #  model settings
    gf_model = get_model_tseries(model_name=args.model_name, num_class=n_class,n_channel=n_channel,
        use_cuda=True, data_parallel=False,dataset=args.dataset)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(gf_model))

    h_model = Projection_TSeries(in_features=gf_model.fc.in_features,
        n_layers=args.n_proj_layer, n_hidden=128).cuda()

    #  training settings
    '''gf_optimizer = torch.optim.SGD(
        gf_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)'''
    gf_optimizer = torch.optim.AdamW(gf_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #follow ptbxl batchmark!!!
    '''scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gf_optimizer,
        float(args.epochs), eta_min=args.learning_rate_min)'''
    scheduler = torch.optim.lr_scheduler.OneCycleLR(gf_optimizer, max_lr=args.learning_rate, epochs = args.epochs, steps_per_epoch = len(train_queue)) #follow ptbxl batchmark!!!

    h_optimizer = torch.optim.Adam(
        h_model.parameters(),
        lr=args.proj_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.proj_weight_decay)

    if not multilabel:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = criterion.cuda()

    #  AdaAug settings
    after_transforms = train_queue.dataset.after_transforms
    adaaug_config = {'sampling': 'prob',
                    'k_ops': 1, #as paper
                    'delta': 0.0, 
                    'temp': 1.0, 
                    'search_d': get_dataset_dimension(args.dataset),
                    'target_d': get_dataset_dimension(args.dataset),
                    'gf_model_name': args.model_name}

    adaaug = AdaAug_TS(after_transforms=after_transforms,
        n_class=n_class,
        gf_model=gf_model,
        h_model=h_model,
        save_dir=args.save,
        config=adaaug_config,
        multilabel=multilabel)
    #for valid data select
    best_val_acc,best_gf,best_h = 0,None,None
    #  Start training
    start_time = time.time()
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        step_dic={'epoch':epoch}
        # searching
        train_acc, train_obj, train_dic = train(train_queue, search_queue, gf_model, adaaug,
            criterion, gf_optimizer, args.grad_clip, h_optimizer, epoch, args.search_freq, multilabel=multilabel,n_class=n_class)

        # validation
        valid_acc, valid_obj,valid_dic = infer(valid_queue, gf_model, criterion, multilabel=multilabel,n_class=n_class,mode='valid')
        logging.info(f'train_acc {train_acc} valid_acc {valid_acc}')
        scheduler.step()
        #test
        test_acc, test_obj, test_dic  = infer(test_queue, gf_model, criterion, multilabel=multilabel,n_class=n_class,mode='test')
        logging.info('test_acc %f', test_acc)
        #val select
        if args.valselect and valid_acc>best_val_acc:
            best_val_acc = valid_acc
            result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            valid_dic['best_valid_acc_avg'] = valid_acc
            test_dic['best_test_acc_avg'] = test_acc
            best_gf = gf_model
            best_h = h_model
        elif not args.valselect:
            best_gf = gf_model
            best_h = h_model

        utils.save_model(best_gf, os.path.join(args.save, 'gf_weights.pt'))
        utils.save_model(best_h, os.path.join(args.save, 'h_weights.pt'))
        
        step_dic.update(test_dic)
        step_dic.update(train_dic)
        step_dic.update(valid_dic)
        wandb.log(step_dic)

    end_time = time.time()
    elapsed = end_time - start_time

    test_acc, test_obj, test_dic = infer(test_queue, gf_model, criterion, multilabel=multilabel,n_class=n_class,mode='test')
    #wandb
    step_dic.update(test_dic)
    step_dic.update(result_valid_dic)
    step_dic.update(result_test_dic)
    wandb.log(step_dic)
    #save&log
    utils.save_model(gf_model, os.path.join(args.save, 'gf_weights.pt'))
    utils.save_model(h_model, os.path.join(args.save, 'h_weights.pt'))
    adaaug.save_history(class2label)
    figure = adaaug.plot_history()

    logging.info(f'test_acc {test_acc}')
    logging.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logging.info(f'saved to: {args.save}')

def train(train_queue, search_queue, gf_model, adaaug, criterion, gf_optimizer,
            grad_clip, h_optimizer, epoch, search_freq, multilabel=False,n_class=10):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    confusion_matrix = torch.zeros(n_class,n_class)
    preds = []
    targets = []
    total = 0
    for step, (input, seq_len, target) in enumerate(train_queue):
        input = input.float().cuda()
        target = target.cuda()

        # exploitation
        timer = time.time()
        aug_images = adaaug(input, seq_len, mode='exploit')
        aug_images = aug_images.cuda()
        gf_model.train()
        gf_optimizer.zero_grad()
        logits = gf_model(aug_images, seq_len)
        if multilabel:
            loss = criterion(logits, target.float())
        else:
            loss = criterion(logits, target.long())
        loss.backward()
        nn.utils.clip_grad_norm_(gf_model.parameters(), grad_clip)
        gf_optimizer.step()

        #  stats
        n = target.size(0)
        objs.update(loss.detach().item(), n)
        if not multilabel:
            prec1, prec5 = utils.accuracy(logits.detach(), target.detach(), topk=(1, 5))
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)
        exploitation_time = time.time() - timer

        # exploration
        timer = time.time()
        if step % search_freq == 0:
            input_search, seq_len, target_search = next(iter(search_queue))
            input_search = input_search.float().cuda()
            target_search = target_search.cuda()
            gf_optimizer.zero_grad()
            h_optimizer.zero_grad()
            mixed_features = adaaug(input_search, seq_len, mode='explore')
            logits_search = gf_model.classify(mixed_features)
            if multilabel:
                loss = criterion(logits_search, target_search.float())
            else:
                loss = criterion(logits_search, target_search.long())
            
            loss.backward()
            h_optimizer.step()
            exploration_time = time.time() - timer
            gf_optimizer.zero_grad()
            h_optimizer.zero_grad()

            #  log policy
            adaaug.add_history(input_search, seq_len, target_search)

        global_step = epoch * len(train_queue) + step
        if global_step % args.report_freq == 0:
            logging.info('  |train %03d %e %f %f | %.3f + %.3f s', global_step,
                objs.avg, top1.avg, top5.avg, exploitation_time, exploration_time)
        #ACC-AUROC
        if not multilabel:
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            for t, p in zip(target.data.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        else:
            predicted = torch.sigmoid(logits)
        preds.append(predicted.cpu().detach())
        targets.append(target.cpu().detach())
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
    out_dic[f'train_loss'] = objs.avg
    if multilabel:
        ptype = 'auroc'
    else:
        ptype = 'acc'
    out_dic[f'train_{ptype}_avg'] = perfrom
    for i,e_c in enumerate(perfrom_cw):
        out_dic[f'train_{ptype}_c{i}'] = e_c

    return perfrom, objs.avg, out_dic


def infer(valid_queue, gf_model, criterion, multilabel=False, n_class=10,mode='test'):
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


if __name__ == '__main__':
    main()
