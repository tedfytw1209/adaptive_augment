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
from dataset import get_num_class, get_dataloaders, get_label_name, get_dataset_dimension, get_ts_dataloaders
from config import get_warmup_config
from warmup_scheduler import GradualWarmupScheduler
import wandb

parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--dataroot', type=str, default='./', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--default_split', action='store_true', help='use dataset deault split')
parser.add_argument('--test_size', type=float, default=0.2, help='test size ratio (if needed)')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--multilabel', action='store_true', default=False, help='using multilabel default False')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--use_parallel', action='store_true', default=False, help="use data parallel default False")
parser.add_argument('--model_name', type=str, default='wresnet40_2', help="model name")
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--epochs', type=int, default=600, help='number of training epochs')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--search_dataset', type=str, default='./', help='search dataset name')
parser.add_argument('--gf_model_name', type=str, default='./', help='gf_model name')
parser.add_argument('--gf_model_path', type=str, default='./', help='gf_model path')
parser.add_argument('--h_model_path', type=str, default='./', help='h_model path')
parser.add_argument('--k_ops', type=int, default=1, help="number of augmentation applied during training")
parser.add_argument('--delta', type=float, default=0.3, help="degree of perturbation in magnitude")
parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of additional hidden layer in augmentation policy projection")
parser.add_argument('--n_proj_hidden', type=int, default=128, help="number of hidden units in augmentation policy projection layers")
parser.add_argument('--restore_path', type=str, default='./', help='restore model path')
parser.add_argument('--restore', action='store_true', default=False, help='restore model default False')

args = parser.parse_args()
debug = True if args.save == "debug" else False
args.save = '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save)
if debug:
    args.save = os.path.join('debug', args.save)
else:
    args.save = os.path.join('eval', args.dataset, args.save)
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

    #  dataset settings
    n_class = get_num_class(args.dataset)
    class2label = get_label_name(args.dataset, args.dataroot)
    multilabel = args.multilabel
    train_queue, valid_queue, _, test_queue = get_ts_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        args.dataroot, args.cutout, args.cutout_length,
        split=args.train_portion, split_idx=0, target_lb=-1,
        search=True, test_size=args.test_size,multilabel=args.multilabel)

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'  |total: {len(train_queue.dataset)}')
    logging.info(f'  |train: {len(train_queue)*args.batch_size}')
    logging.info(f'  |valid: {len(valid_queue)*args.batch_size}')

    #  task model settings
    task_model = get_model_tseries(model_name=args.model_name,
                            num_class=n_class,
                            use_cuda=True, data_parallel=False,dataset=args.dataset)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(task_model))

    #  task optimization settings
    optimizer = torch.optim.SGD( #paper not mention
        task_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    m, e = get_warmup_config(args.dataset)
    scheduler = GradualWarmupScheduler( #paper not mention
            optimizer,
            multiplier=m,
            total_epoch=e,
            after_scheduler=scheduler)
    logging.info(f'Optimizer: SGD, scheduler: CosineAnnealing, warmup: {m}/{e}')
    if not multilabel:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduce='mean')
    criterion = criterion.cuda()

    #  restore setting
    if args.restore:
        trained_epoch = utils.restore_ckpt(task_model, optimizer, scheduler, args.restore_path, location=args.gpu) + 1
        n_epoch = args.epochs - trained_epoch
        logging.info(f'Restoring model from {args.restore_path}, starting from epoch {trained_epoch}')
    else:
        trained_epoch = 0
        n_epoch = args.epochs

    #  load trained adaaug sub models
    search_n_class = get_num_class(args.search_dataset)
    gf_model = get_model_tseries(model_name=args.gf_model_name,
                            num_class=search_n_class,
                            use_cuda=True, data_parallel=False,dataset=args.dataset)

    h_model = Projection_TSeries(in_features=gf_model.fc.in_features,
                            n_layers=args.n_proj_layer,
                            n_hidden=args.n_proj_hidden).cuda()

    utils.load_model(gf_model, f'{args.gf_model_path}/gf_weights.pt', location=args.gpu)
    utils.load_model(h_model, f'{args.h_model_path}/h_weights.pt', location=args.gpu)

    for param in gf_model.parameters():
        param.requires_grad = False

    for param in h_model.parameters():
        param.requires_grad = False

    after_transforms = train_queue.dataset.after_transforms
    adaaug_config = {'sampling': 'prob',
                    'k_ops': args.k_ops,
                    'delta': args.delta,
                    'temp': args.temperature,
                    'search_d': get_dataset_dimension(args.search_dataset),
                    'target_d': get_dataset_dimension(args.dataset)}

    adaaug = AdaAug_TS(after_transforms=after_transforms,
                    n_class=search_n_class,
                    gf_model=gf_model,
                    h_model=h_model,
                    save_dir=args.save,
                    config=adaaug_config)

    #  start training
    for i_epoch in range(n_epoch):
        epoch = trained_epoch + i_epoch
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj = train(
            train_queue, task_model, criterion, optimizer, epoch, args.grad_clip, adaaug, multilabel=multilabel)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj, _, _ = infer(valid_queue, task_model, criterion, multilabel=multilabel)
        logging.info('valid_acc %f', valid_acc)

        scheduler.step()

        if epoch % args.report_freq == 0:
            test_acc, test_obj, test_acc5, _ = infer(test_queue, task_model, criterion, multilabel=multilabel)
            logging.info('test_acc %f %f', test_acc, test_acc5)

        utils.save_ckpt(task_model, optimizer, scheduler, epoch,
            os.path.join(args.save, 'weights.pt'))

    adaaug.save_history(class2label)
    figure = adaaug.plot_history()
    test_acc, test_obj, test_acc5, _ = infer(test_queue, task_model, criterion, multilabel=multilabel)

    logging.info('test_acc %f %f', test_acc, test_acc5)
    logging.info(f'save to {args.save}')


def train(train_queue, model, criterion, optimizer, epoch, grad_clip, adaaug, multilabel=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    preds = []
    targets = []
    total = 0
    for step, (input, seq_len, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        #  get augmented training data from adaaug
        aug_images = adaaug(input, seq_len, mode='exploit')
        model.train()
        optimizer.zero_grad()
        logits = model(aug_images, seq_len)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
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
            adaaug.add_history(input, target)
        
        # Accuracy / AUROC
        if not multilabel:
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
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
    
    return perfrom, objs.avg


def infer(valid_queue, model, criterion, multilabel=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    confusion_matrix = torch.zeros(10,10)
    preds = []
    targets = []
    total = 0
    with torch.no_grad():
        for input,seq_len, target in valid_queue:
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input,seq_len)
            loss = criterion(logits, target)
            
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
        perfrom = top1.avg * 100
        perfrom2 = top5.avg * 100
    else:
        targets_np = torch.cat(targets).numpy()
        preds_np = torch.cat(preds).numpy()
        perfrom_cw = utils.AUROC_cw(targets_np,preds_np)
        perfrom = perfrom_cw.mean()
        perfrom2 = perfrom
        logging.info('class-wise AUROC: ' + '['+', '.join(['%.1f'%e for e in perfrom_cw])+']')
        logging.info('Overall AUROC: %f',perfrom)
    return perfrom, objs.avg, perfrom2, objs.avg

if __name__ == '__main__':
    main()
