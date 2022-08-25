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
from step_function import train,infer
from warmup_scheduler import GradualWarmupScheduler
import wandb

parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--dataroot', type=str, default='./', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
parser.add_argument('--labelgroup', default='')
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
parser.add_argument('--valselect', action='store_true', default=False, help='use valid select')
parser.add_argument('--notwarmup', action='store_true', default=False, help='use valid select')
parser.add_argument('--augselect', type=str, default='', help="augmentation selection")
parser.add_argument('--diff_aug', action='store_true', default=False, help='use valid select')
parser.add_argument('--not_reweight', action='store_true', default=False, help='use valid select')
parser.add_argument('--lambda_aug', type=float, default=1.0, help="augment sample weight")
parser.add_argument('--class_adapt', action='store_true', default=False, help='class adaptive')
parser.add_argument('--class_embed', action='store_true', default=False, help='class embed') #tmp use
parser.add_argument('--loss_type', type=str, default='minus', help="loss type for difficult policy training", choices=['minus','relative','adv'])
parser.add_argument('--keep_aug', action='store_true', default=False, help='info keep augment')
parser.add_argument('--keep_mode', type=str, default='auto', help='info keep mode',choices=['auto','b','p','t'])
parser.add_argument('--keep_thres', type=float, default=0.6, help="augment sample weight")

args = parser.parse_args()
debug = True if args.save == "debug" else False
if args.k_ops>0:
    Aug_type = 'AdaAug'
else:
    Aug_type = 'NOAUG'
now_str = time.strftime("%Y%m%d-%H%M%S")
args.save = '{}-{}'.format(now_str, args.save)
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
    date_time_str = now_str
    h_model_dir = args.h_model_path
    h_model_dir = h_model_dir.strip('/').split('/')[-1]
    #wandb
    group_name = f'{Aug_type}_tottrain{args.augselect}_vselect_{args.dataset}{args.labelgroup}_{args.model_name}_hmodel{h_model_dir}_e{args.epochs}_lr{args.learning_rate}'
    experiment_name = f'{Aug_type}{args.k_ops}_tottrain{args.augselect}_vselect_{args.dataset}{args.labelgroup}_{args.model_name}_hmodel{h_model_dir}_e{args.epochs}_lr{args.learning_rate}'
    run_log = wandb.init(config=args, 
                  project='AdaAug',
                  group=group_name,
                  name=f'{date_time_str}_' + experiment_name,
                  dir='./',
                  job_type="DataAugment",
                  reinit=True)

    #  dataset settings
    n_channel = get_num_channel(args.dataset)
    n_class = get_num_class(args.dataset,args.labelgroup)
    class2label = get_label_name(args.dataset, args.dataroot)
    multilabel = args.multilabel
    diff_augment = args.diff_aug
    diff_reweight = not args.not_reweight
    train_queue, valid_queue, _, test_queue,_ = get_ts_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        args.dataroot, args.cutout, args.cutout_length,
        split=args.train_portion, split_idx=0, target_lb=-1,
        search=False, test_size=args.test_size,multilabel=args.multilabel,
        default_split=args.default_split,labelgroup=args.labelgroup)

    logging.info(f'Dataset: {args.dataset}')
    logging.info(f'  |total: {len(train_queue.dataset)}')
    logging.info(f'  |train: {len(train_queue)*args.batch_size}')
    logging.info(f'  |valid: {len(valid_queue)*args.batch_size}')

    #  task model settings
    task_model = get_model_tseries(model_name=args.model_name,
                            num_class=n_class,n_channel=n_channel,
                            use_cuda=True, data_parallel=False,dataset=args.dataset)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(task_model))

    #  task optimization settings
    '''optimizer = torch.optim.SGD( #paper not mention
        task_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
        )'''
    #follow ptbxl batchmark!!!
    optimizer = torch.optim.AdamW(task_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    '''scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)'''
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, epochs = args.epochs, steps_per_epoch = len(train_queue)) #follow ptbxl batchmark!!!
    if not args.notwarmup:
        m, e = get_warmup_config(args.dataset)
        scheduler = GradualWarmupScheduler( #paper not mention!!!
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
    search_n_class = get_num_class(args.search_dataset,args.labelgroup)
    gf_model = get_model_tseries(model_name=args.gf_model_name,
                            num_class=search_n_class,n_channel=n_channel,
                            use_cuda=True, data_parallel=False,dataset=args.dataset)
    h_input = gf_model.fc.in_features
    if args.class_adapt:
        h_input += n_class
    h_model = Projection_TSeries(in_features=h_input,
                            n_layers=args.n_proj_layer,
                            n_hidden=args.n_proj_hidden,
                            augselect=args.augselect).cuda()

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
                    'target_d': get_dataset_dimension(args.dataset),
                    'gf_model_name': args.gf_model_name}
    keepaug_config = {'keep_aug':args.keep_aug,'mode':args.keep_mode,'thres':args.keep_thres,'length':200} #tmp!!!
    adaaug = AdaAug_TS(after_transforms=after_transforms,
                    n_class=search_n_class,
                    gf_model=gf_model,
                    h_model=h_model,
                    save_dir=args.save,
                    config=adaaug_config,
                    keepaug_config=keepaug_config,
                    multilabel=multilabel,
                    augselect=args.augselect,
                    class_adaptive=args.class_adapt)
    #for valid data select
    best_val_acc,best_task = 0,None
    result_valid_dic, result_test_dic = {}, {}
    #  start training
    for i_epoch in range(n_epoch):
        epoch = trained_epoch + i_epoch
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        step_dic = {'epoch':epoch}
        diff_dic = {'difficult_aug':diff_augment,'reweight':diff_reweight,'lambda_aug':args.lambda_aug, 'class_adaptive':args.class_adapt,
                }
        train_acc, train_obj, train_dic = train(args,
            train_queue, task_model, criterion, optimizer, scheduler, epoch, args.grad_clip, adaaug, 
                multilabel=multilabel,n_class=n_class,**diff_dic)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj, _, _, valid_dic = infer(valid_queue, task_model, criterion, multilabel=multilabel,n_class=n_class,mode='valid')
        logging.info('valid_acc %f', valid_acc)
        test_acc, test_obj, test_acc5, _,test_dic  = infer(test_queue, task_model, criterion, multilabel=multilabel,n_class=n_class,mode='test')
        logging.info('test_acc %f %f', test_acc, test_acc5)
        #scheduler.step()
        #val select
        if args.valselect and valid_acc>best_val_acc:
            best_val_acc = valid_acc
            result_valid_dic = {f'result_{k}': valid_dic[k] for k in valid_dic.keys()}
            result_test_dic = {f'result_{k}': test_dic[k] for k in test_dic.keys()}
            valid_dic['best_valid_acc_avg'] = valid_acc
            test_dic['best_test_acc_avg'] = test_acc
            best_task = task_model
        elif not args.valselect:
            best_task = task_model
        #wandb
        step_dic.update(train_dic)
        step_dic.update(valid_dic)
        step_dic.update(test_dic)

        utils.save_ckpt(best_task, optimizer, scheduler, epoch,
            os.path.join(args.save, 'weights.pt'))
        #wandb log
        wandb.log(step_dic)

    adaaug.save_history(class2label)
    figure = adaaug.plot_history()
    test_acc, test_obj, test_acc5, _, test_dic = infer(test_queue, task_model, criterion, multilabel=multilabel,n_class=n_class,mode='test')
    #wandb
    step_dic.update(test_dic)
    step_dic.update(result_valid_dic)
    step_dic.update(result_test_dic)
    wandb.log(step_dic)
    logging.info('test_acc %f %f', test_acc, test_acc5)
    logging.info(f'save to {args.save}')

if __name__ == '__main__':
    main()
